from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn
import numpy as np

class ConvertToBGR:
    """
    Converts a PIL image from RGB to BGR
    """

    def __init__(self):
        pass

    def __call__(self, img: Image):
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        return img

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class Multiplier:
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, img):
        return img*self.multiple

    def __repr__(self):
        return "{}(multiple={})".format(self.__class__.__name__, self.multiple)


class TrainTransform(nn.Module):
    def __init__(self):
        super(TrainTransform, self).__init__()

        self.converter = ConvertToBGR()
        self.resizer = transforms.Resize(256)
        self.multiplier = Multiplier(255)
        self.normalizer = transforms.Normalize(mean = [104, 117, 128], 
                                               std = [1, 1, 1])

    def forward(self, image: Image, mask: Image):
        # convert to bgr
        image = self.converter(image)

        # resize
        image = self.resizer(image)
        mask = self.resizer(mask)

        # crop and resize
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.16, 1), ratio=(0.75, 1.33))
        image = TF.resized_crop(image, i, j, h, w, size=(227, 227))
        mask = TF.resized_crop(mask, i, j, h, w, size=(227, 227))

        # flip
        if np.random.rand() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # multiply
        image = self.multiplier(image)

        # normalize
        image = self.normalizer(image)

        return image, mask


class ValTransform(nn.Module):
    def __init__(self):
        super(ValTransform, self).__init__()

        self.converter = ConvertToBGR()
        self.resizer = transforms.Resize(256)
        self.multiplier = Multiplier(255)
        self.normalizer = transforms.Normalize(mean = [104, 117, 128], 
                                               std = [1, 1, 1])

    def forward(self, image, mask):
        # convert to bgr
        image = self.converter(image)

        # resize
        image = self.resizer(image)
        mask = self.resizer(mask)

        # crop and resize
        image = TF.center_crop(image, (227, 227))
        mask = TF.center_crop(mask, (227, 227))

        # convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # multiply
        image = self.multiplier(image)

        # normalize
        image = self.normalizer(image)

        return image, mask


def inv_transform(image):
    image = transforms.Normalize(
        mean=[-104, -117, -128],
        std=[256, 256, 256]
    )(image)

    image = image.flip(0)

    return image
