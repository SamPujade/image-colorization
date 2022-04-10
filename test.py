import torch
import torch.nn as nn
import torch.utils.data as data_utils
import yaml
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt
from PIL import Image

from dataset import ImageDataset
from networks.generator import Generator
from utils.lab_to_rgb import lab_to_rgb
from utils.build_res_unet import build_res_unet


def test(params):
    # Initialize generator and loss criterion
    G = Generator()
    l_criterion = nn.L1Loss()

    # We use GPU if available, otherwise CPU
    if torch.cuda.is_available():
        print("Using GPU with cuda")
    else:
        print("Using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load generator
    if params.test.pretrained:
        G = build_res_unet(n_input=1, n_output=2, size=256)
    else:
        G = Generator()
    if "G" in params.test:
        G.load_state_dict(torch.load(params.test.G, map_location=device))
    G = G.to(device)

    # Load dataset
    dataset = ImageDataset(params.dataset.test_root_dir, params.dataset.size, params.dataset.n_test_images)
    train_loader = data_utils.DataLoader(dataset, shuffle=True, num_workers=1)

    running_loss = 0.0
    num_steps = 0

    for i, data in enumerate(train_loader):
        # Split data into channels
        num_steps += 1
        real_l_data = data[:, 0:1, :, :].to(device) / 50 - 1
        real_ab_data = data[:, 1:, :, :].to(device) / 128
        fake_ab_data = G(real_l_data)

        # Plot original image, grayscale image and generated image
        if i == int(params.test.show_index):
            np_fake_img = lab_to_rgb(real_l_data, fake_ab_data)[0]
            np_real_img = lab_to_rgb(real_l_data, real_ab_data)[0]
            np_bw_img = lab_to_rgb(real_l_data, torch.zeros(1, 2, data.shape[2], data.shape[3]))[0]

            fig = plt.gcf()
            #fig.canvas.set_window_title('Colorize image')

            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(np_real_img)
            ax1.set_title('Original Image')
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(np_bw_img)
            ax2.set_title('Black and White Image')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(np_fake_img)
            ax3.set_title('Generated Image')

            plt.show()

        # Save original and generated images
        if i < params.test.save_nb:
            np_fake_img = lab_to_rgb(real_l_data, fake_ab_data)[0]
            np_real_img = lab_to_rgb(real_l_data, real_ab_data)[0]

            im = Image.fromarray((np_fake_img * 255).astype(np.uint8))
            im = im.save(f"{params.test.save_path}/test_{i:04d}.png")
            im2 = Image.fromarray((np_real_img * 255).astype(np.uint8))
            im2 = im2.save(f"{params.test.save_path}/test_{i:04d}_real.png")

        # Calculate the image distance loss pixelwise between the images.
        loss = l_criterion(real_ab_data, fake_ab_data)
        running_loss += loss.item()
    
    print('Mean Absolute Error(MAE): ', np.round(running_loss / num_steps, 5))

if __name__ == '__main__':

    # Extract parameters
    with open('conf/params.yml', 'r') as stream:
        try:
            args = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    test(AttrDict(args))
