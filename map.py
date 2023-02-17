import cv2
import sys
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
from DCGAN_inference import Generator, Discriminator


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_netG = torch.load("new_nudes/DCGAN_netG.pt", map_location=device)
    model_netG.eval()
    
    result = result_map(5, 5, 64, 0.4)
    uint_result = result.astype(np.uint8) * 255
    cv2.imwrite("result_map.png", uint_result)


if __name__ == "__main__":
    model_netG = torch.load("new_nudes/DCGAN_netG.pt", map_location=torch.device("cpu"))
    model_netG.eval()
    sys.exit(main())
