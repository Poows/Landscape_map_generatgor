import cv2
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Root directory for dataset
dataroot = "drive/MyDrive/VKR/dataset/png_images_class/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == "cuda") and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
# netG.apply(weights_init)

# Print the model
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == "cuda") and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

model_netG = torch.load("new_nudes/DCGAN_netG.pt", map_location=torch.device("cpu"))
model_netG.eval()

fixed_noise_test = torch.rand(1, 100, 1, 1, device="cpu")
fake_model_netG = model_netG(fixed_noise_test).detach().cpu().numpy()
fake_img = np.transpose(fake_model_netG.reshape(3, 64, 64), (1, 2, 0))


def get_preprocess_image(TensorImage, ReturnGray):
    fake_img = np.transpose(
        vutils.make_grid(TensorImage, normalize=True, padding=0)
        .detach()
        .cpu()
        .numpy()
        .reshape(3, 64, 64),
        (1, 2, 0),
    )
    fake_img[fake_img < 0.0] = 0.0
    fake_img = (fake_img * 255).astype(np.uint8)
    if ReturnGray:
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2GRAY)
    return fake_img


def generate_image():
    fixed_noise_test = torch.rand(1, 100, 1, 1, device="cpu")
    generated_image = get_preprocess_image(model_netG(fixed_noise_test), True)
    return generated_image


# import time

# def get_x_maps(image_map1, image_map2, width):
#   #merge_size = round((image_map1.shape[0] * procent), 0)

#   image_map1 = image_map1.astype(np.float32)
#   image_map2 = image_map2.astype(np.float32)

#   coef = 1

#   start = int(image_map1.shape[1] - width)
#   #width = image_map1.shape[0] - start

#   for iter in range(start, image_map1.shape[1]):
#     coef = 1. - float(iter - start) / width
#     image_map1[:, iter] = image_map1[:, iter] * coef
#     image_map2[:, iter - start] = image_map2[:, iter - start] * (1 - coef)

#   return image_map1, image_map2#, widths

# def merge_x_maps(image_map1, image_map2, width):
#   result = np.zeros((int(image_map1.shape[0]), int(image_map1.shape[1] + image_map2.shape[1] - width)))
#   result[:, 0:image_map1.shape[1]] += image_map1

#   start = int(image_map1.shape[1] - width)

#   for iter in range(start, image_map1.shape[1]):
#       result[:, iter] = (result[:, iter] + image_map2[:, iter - start])
#   result[:, image_map1.shape[1]:result.shape[1]] += image_map2[:, width:image_map2.shape[1]]
#   return result


def get_y_maps(image_map1, image_map2, width):
    coef = 1

    if image_map1 is not None:
        image_map1 = image_map1.astype(np.float32)
        imap1 = image_map1.shape[0]
        start = int(imap1 - width)
        for iter in range(start, imap1):
            coef = 1.0 - float(iter - start) / width
            image_map1[iter, :] = image_map1[iter, :] * coef

    coef = 1

    if image_map2 is not None:
        image_map2 = image_map2.astype(np.float32)
        imap2 = image_map2.shape[0]
        start = int(imap2 - width)
        for iter in range(start, imap2):
            coef = 1.0 - float(iter - start) / width
            image_map2[iter - start, :] = image_map2[iter - start, :] * (1 - coef)

    return image_map1, image_map2


def get_x_maps(image_map1, image_map2, width):
    image_map1 = image_map1.astype(np.float32)
    image_map2 = image_map2.astype(np.float32)

    coef = 1

    start = int(image_map1.shape[1] - width)

    for iter in range(start, image_map1.shape[1]):
        coef = 1.0 - float(iter - start) / width
        image_map1[:, iter] = image_map1[:, iter] * coef
        image_map2[:, iter - start] = image_map2[:, iter - start] * (1 - coef)

    return image_map1, image_map2


def merge_x_maps(image_map1, image_map2, width):
    result = np.zeros(
        (
            int(image_map1.shape[0]),
            int(image_map1.shape[1] + image_map2.shape[1] - width),
        )
    )
    result[:, 0 : image_map1.shape[1]] += image_map1

    start = int(image_map1.shape[1] - width)

    for iter in range(start, image_map1.shape[1]):
        result[:, iter] = result[:, iter] + image_map2[:, iter - start]
    result[:, image_map1.shape[1] : result.shape[1]] += image_map2[
        :, width : image_map2.shape[1]
    ]
    return result


def merge_y_maps(image_map1, image_map2, width):
    result = np.zeros(
        (
            int(image_map1.shape[0] + image_map2.shape[0] - width),
            int(image_map1.shape[1]),
        )
    )
    result[0 : image_map1.shape[0], :] += image_map1

    start = int(image_map1.shape[0] - width)

    for iter in range(start, image_map1.shape[0]):
        result[iter, :] = result[iter, :] + image_map2[iter - start, :]
    result[image_map1.shape[0] : result.shape[0], :] += image_map2[
        width : image_map2.shape[0], :
    ]
    return result


def result_map(map_x_size, map_y_size, image_size, procent):
    width = int(round((image_size * procent), 0))
    for y in range(map_y_size):
        for x in range(map_x_size):
            if x == 0:
                map1, map2 = get_x_maps(generate_image(), generate_image(), width)
                result = merge_x_maps(map1, map2, width)
            else:
                result, map2 = get_x_maps(result, generate_image(), width)
                result = merge_x_maps(result, map2, width)
        if y % 2 == 0:
            if y == 0:
                result_map, _ = get_y_maps(result, None, width)
            else:
                result_map, r1 = get_y_maps(result_map, result, width)
                result_map = merge_y_maps(result_map, r1, width)
        else:
            result_map, r2 = get_y_maps(result_map, result, width)
            result_map = merge_y_maps(result_map, r2, width)
    return result_map


def main():
    result = result_map(5, 5, 64, 0.4)
    uint_result = result.astype(np.uint8) * 255
    cv2.imwrite("result_map.png", uint_result)


if __name__ == "__main__":
    sys.exit(main())
