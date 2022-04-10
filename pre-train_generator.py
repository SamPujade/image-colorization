from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

import torch
import torch.optim as optim
import torch.nn as nn
from dataset import ImageDataset
from attrdict import AttrDict
import torch.utils.data as data_utils
import yaml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.lab_to_rgb import lab_to_rgb
from utils.build_res_unet import build_res_unet

def pretrain_generator(net_G, train_loader, opt, criterion, epochs):
    # We use GPU if available, otherwise CPU
    if torch.cuda.is_available():
        print("Using GPU with cuda")
    else:
        print("Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for e in range(epochs):
        # The generator loss is summed for the entire epoch.
        g_running_loss = 0.0

        for i, data in enumerate(train_loader):
            print(f"Batch : {i}/{len(dataset)//params.train.batch_size}")
            torch.save(net_G.state_dict(), "models/res18-unet-to-device.pt")

            real_l_data = data[:, 0:1, :, :].to(device) / 50 - 1
            real_ab_data = data[:, 1:, :, :].to(device) / 128
            preds = net_G(real_l_data)
            loss = criterion(preds, real_ab_data) 
            opt.zero_grad()
            loss.backward()
            opt.step()

            g_running_loss += loss.detach().numpy()

            # Save images from pretraining
            # if i % 10 == 0:
            #     np_fake_img = lab_to_rgb(real_l_data, preds)[0]

            #     im = Image.fromarray((np_fake_img * 255).astype(np.uint8))
            #     im = im.save(f"results/pretrain_gen_PIL_{e:02d}_{i:04d}.png")
        
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {g_running_loss:.5f}")


if __name__ == '__main__':

    # Extract parameters
    with open('conf/params.yml', 'r') as stream:
        try:
            args = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    params = AttrDict(args)

    # Load dataset
    dataset = ImageDataset(params.dataset.train_root_dir, params.dataset.size, params.dataset.n_train_images)
    train_loader = data_utils.DataLoader(dataset, batch_size=params.train.batch_size, shuffle=True, num_workers=1)

    print("Pre-training resunet18 model")

    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()        
    pretrain_generator(net_G, train_loader, opt, criterion, 20)
    torch.save(net_G.state_dict(), "models/res18-unet-to-device.pt")