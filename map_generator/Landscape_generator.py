import torch
import numpy as np
import cv2
from pydantic import BaseModel
import torchvision.utils as vutils
from DCGAN_inference import Generator
from configs import StitchingConfig


class GeneratorConfig(BaseModel):
    path_to_save_model: str


class LandscapeGenerator:
    
    generator: Generator
    stitching_method: StitchingConfig
    
    def __init__(self, Generator: GeneratorConfig, config):
        self.generator = torch.load(Generator.path_to_save_model, map_location=torch.device("cpu"))
        self.stitching_method = config.stitching_method
        self.device = config.device

    def get_preprocess_image(self, TensorImage, ReturnGray, shape=(64, 64)):
        fake_img = np.transpose(
            vutils.make_grid(TensorImage, normalize=True, padding=0)
            .detach()
            .cpu()
            .numpy()
            .reshape(3, shape[0], shape[1]),
            (1, 2, 0),
        )
        fake_img[fake_img < 0.0] = 0.0
        fake_img = (fake_img * 255).astype(np.uint8)
        if ReturnGray:
            fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2GRAY)
        return fake_img
    
    def generate_images(self, images_count, image_shape=(64, 64)):
        fixed_noise_test = torch.rand(images_count, 100, 1, 1, device=self.device)
        generated_images = self.get_preprocess_image(self.generator(fixed_noise_test), True, image_shape)
        return generated_images

    def generate_map(self):
        pass