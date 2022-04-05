from PIL import Image
from glob import glob
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from skimage.color import rgb2lab, lab2rgb

class ImageDataset():
    def __init__(self, path, size, n_images=-1):
        self.images = glob(path)
        if n_images > -1:
            self.images = self.images[:n_images]
        self.transforms = transforms.Compose([
                transforms.Resize((size, size),  transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = self.transforms(image)
        image_lab = rgb2lab(np.array(image)).astype("float32") # Converting RGB to L*a*b
        image_lab = transforms.ToTensor()(image_lab)
        return image_lab