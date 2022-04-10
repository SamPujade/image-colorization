import torch
import torch.nn as nn
import torch.utils.data as data_utils
import yaml
from attrdict import AttrDict
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from dataset import ImageDataset
from networks.generator import Generator
from networks.discriminator import Discriminator
from utils.lab_to_rgb import lab_to_rgb
from utils.build_res_unet import build_res_unet
from utils.remaining_time import remaining_time


def weights_init(m, gain=0.02):
    classname = m.__class__.__name__

    if hasattr(m, 'weight') and 'Conv' in classname:
        nn.init.normal_(m.weight.data, mean=0.0, std=gain)
    elif 'BatchNorm2d' in classname:
        nn.init.normal_(m.weight.data, 1., gain)
        nn.init.constant_(m.bias.data, 0.)

    return


def train(params):

    # We use GPU if available, otherwise CPU
    if torch.cuda.is_available():
        print("Using GPU with cuda")
    else:
        print("Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generator initialization
    # Use pre-trained model
    if params.train.use_pretrain:
        print(f"Using pre-trained model from {params.train.pretrain_model}")
        G = build_res_unet(n_input=1, n_output=2, size=256)
        G.load_state_dict(torch.load(params.train.pretrain_model, map_location=device))
        G = G.to(device)
    else:
        G = Generator().to(device)
        G.apply(weights_init)
    
    # Discriminator initialization
    D = Discriminator().to(device)
    D.apply(weights_init)
    print("Model initialized")

    # Load models to restore
    if "G" in params.restore:
        G.load_state_dict(torch.load(params.restore.G, map_location=device))
    if "D" in params.restore:
        D.load_state_dict(torch.load(params.restore.D, map_location=device))

    # Initialize loss functions and optimizer
    GANcriterion = nn.BCELoss().to(device)
    L1criterion = nn.L1Loss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=params.train.d_learning_rate, betas=(params.train.beta1, params.train.beta2))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=params.train.g_learning_rate, betas=(params.train.beta1, params.train.beta2))

    # Load dataset
    dataset = ImageDataset(params.dataset.train_root_dir, params.dataset.size, params.dataset.n_train_images)
    train_loader = data_utils.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=True, num_workers=1)
    print("Dataset loaded")

    t_0 = time.time()
    nb_of_batch = len(dataset)//params.train.batch_size
    d_running_losses = []
    g_running_losses = []

    for epoch in range(params.train.epochs):
        print(f"\nEpoch : {epoch+1}/{params.train.epochs}")

        # The generator and discriminator losses are summed for the entire epoch.
        d_running_loss = 0.0
        g_running_loss = 0.0

        for i, data in enumerate(train_loader):

            if i >= nb_of_batch:
                break

            # Print estimated remaining time of training
            rem_time = remaining_time(t_0, epoch, params.train.epochs, i, nb_of_batch)
            print(f"Batch : {i+1}/{nb_of_batch} ; estimated remaining time : {rem_time}", end="\r")
            
            # Split real images into channels, generate fake images
            real_l_data = data[:, 0:1, :, :].to(device) / 50 - 1
            real_ab_data = data[:, 1:, :, :].to(device) / 128
            real_lab_data = torch.cat([real_l_data, real_ab_data], 1)
            fake_ab_data = G(real_l_data)
            fake_data = torch.cat([real_l_data, fake_ab_data], 1)

            for p in D.parameters():
                p.requires_grad = True

            # Train the discriminator on real data
            d_optimizer.zero_grad()
            d_real_decision = D(real_lab_data)
            d_real_loss = GANcriterion(d_real_decision, 0.9 * torch.ones(params.train.batch_size).to(device))

            # Train the discriminator on fake data
            d_fake_decision = D(fake_data)
            d_fake_loss = GANcriterion(d_fake_decision, torch.ones(params.train.batch_size).to(device))

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            for p in D.parameters():
                p.requires_grad = False

            # Train the generator
            g_optimizer.zero_grad()
            d_fake_decision = D(fake_data)
            g_fake_loss =  GANcriterion(d_fake_decision, torch.ones(params.train.batch_size).to(device))

            g_image_distance_loss = 100 * L1criterion(fake_ab_data, real_ab_data)
            g_loss = g_fake_loss + g_image_distance_loss
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

            # Update running losses
            d_running_loss += d_loss.detach().cpu().numpy()
            g_running_loss += g_loss.detach().cpu().numpy()
            
            # Print running losses every 10 iterations
            if i % 10 == 0:
                print('[%d, %5d] d_loss: %.5f g_loss: %.5f' %
                        (epoch + 1, i + 1, d_running_loss / 10, g_running_loss / 10))
                d_running_losses.append(d_running_loss)
                g_running_losses.append(g_running_loss)
                d_running_loss = 0.0
                g_running_loss = 0.0

                # Save images from training

                # np_fake_img = lab_to_rgb(real_l_data, fake_ab_data)[0]
                # np_real_img = lab_to_rgb(real_l_data, real_ab_data)[0]

                # im = Image.fromarray((np_fake_img * 255).astype(np.uint8))
                # im = im.save(f"results/GANtrain/train_GAN_{epoch:02d}_{i:04d}.png")
                # im2 = Image.fromarray((np_real_img * 255).astype(np.uint8))
                # im2 = im2.save(f"results/GANtrain/train_GAN_{epoch:02d}_{i:04d}_real.png")

        # Save the generator and discriminator state after each epoch
        if epoch % params.save.every == 0:
            torch.save(G.state_dict(), params.save.G)
            torch.save(D.state_dict(), params.save.D)

    # Plot and save running losses of the discriminator and generator
    save_name = params.save.G.split('/')[-1]
    X = range(len(d_running_losses))
    plt.plot(X, d_running_losses, label='Discriminator')
    plt.plot(X, g_running_losses, label='Generator')
    plt.xlabel("Running loss")
    plt.ylabel("Iterations")
    plt.legend()
    plt.savefig(f'results/plots/loss_{save_name}')
    

if __name__ == '__main__':

    # Extract parameters
    with open('conf/params.yml', 'r') as stream:
        try:
            args = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    train(AttrDict(args))
