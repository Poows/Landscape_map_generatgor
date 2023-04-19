from configs import UploadConfig
from Landscape_generator import LandscapeGenerator, Generator
from texture_synthesys_imagequilting import quilt
from texture_synthesys_kdtrees import Inpaint
import torchvision.utils as vutils
import cv2
import math
import numpy as np
import argparse
import torch


def get_preprocess_image(TensorImage, ReturnGray, shape=(64, 64), image_type="uint8"):
    fake_img = np.transpose(
        vutils.make_grid(TensorImage, normalize=True, padding=0)
        .detach()
        .cpu()
        .numpy()
        .reshape(3, shape[0], shape[1]),
        (1, 2, 0),
    )
    fake_img[fake_img < 0.0] = 0.0
    if image_type == "uint8":
        fake_img = (fake_img * 255).astype(np.uint8)
    elif image_type == "uint16":
        fake_img = (fake_img * 65535).astype(np.uint16)
    if ReturnGray:
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2GRAY)
    return fake_img
    

def generate_images(model_netG, device, image_shape=(64, 64), latent_vector_size=100):
    fixed_noise_test = torch.rand(1, latent_vector_size, 1, 1, device=device)
    generated_images = get_preprocess_image(model_netG(fixed_noise_test), True, image_shape)
    return generated_images


def generate_maps(model_path, image_shape=(600, 600), map_size=64, image_type="uint8"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_netG = torch.load(model_path, map_location=torch.device(device))
    model_netG.eval()

    h, w = image_shape
    i_steps = math.ceil(h / map_size)
    j_steps = math.ceil(w / map_size)

    if image_type == "uint8":
        result = np.zeros((i_steps * map_size, j_steps * map_size * 4), dtype=np.uint8)
    elif image_type == "uint16":
        result = np.zeros((i_steps * map_size, j_steps * map_size * 4), dtype=np.uint8)

    for i in range(0, i_steps):
        for j in range(0, j_steps * 4, 4):
            gi = generate_images(model_netG, "cpu")
            for k in range(4):
                result[i * map_size:(i + 1) * map_size, (j + k) * map_size:((j + k) + 1) * map_size] = gi
                gi = np.rot90(gi, -1)
    return result


def save_map(result, save_type, name):
    if save_type == 'png':
        result = (result * 255).astype(np.uint8)
        cv2.imwrite((name + ".png"), result)
    elif save_type == 'jpg':
        result = (result * 255).astype(np.uint8)
        cv2.imwrite(name + ".jpg", result)
    elif save_type == 'tif':
        result = (result * 65535).astype(np.uint16)
        cv2.imwrite(name + ".tif", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser = parser.parse_args()

    config_path = parser.config
    config = UploadConfig.parse_file(config_path)
    method = config.stitching_method.method
    model_path = config.load_model_and_stuff.model_path

    if method == 'Quilting':
        coef = config.quilting.coef
        map_size = config.quilting.map_size
        patch_map = map_size[0] * coef, map_size[1]
        patch_size = config.quilting.patch_size
        overlap_part = config.quilting.overlap_part
        texture = generate_maps(model_path, image_shape=patch_map)
        quilting_map = quilt(texture, patch_size, map_size, overlap_part, mode="cut")
        save_map(quilting_map, config.quilting.map_type, name="result_map_quilting")
    elif method == 'Gradient':
        pass
    elif method == 'PatchBased':
        map_size = config.patch_based.map_size
        overlap_part = config.patch_based.overlap_part
        method = config.patch_based.method
        patch_size = config.quilting.patch_size
        overlap_size = math.ceil(overlap_part * patch_size)
        vector_size = config.patch_based.maps_image_vector_size
        maps_image = generate_maps(model_path, image_shape=(vector_size * patch_size, 1))
        gm = Inpaint(maps_image, None, patch_size, overlap_size, map_size, window_step=patch_size, method=method, model_path=model_path)
        patchbased_map = gm.resolve()
        save_map(patchbased_map, config.patch_based.map_type, name="result_map_patchbased")


if __name__ == "__main__":
    main()