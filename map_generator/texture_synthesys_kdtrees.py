import numpy as np
import math
import heapq
import torch
import cv2
import torchvision.utils as vutils
from Landscape_generator import LandscapeGenerator, Generator

from scipy.spatial import cKDTree as KDTree
from skimage.util.shape import view_as_windows
from skimage.filters import gaussian
from skimage.transform import rotate
from skimage import exposure


class Inpaint:
    """
    The Inpaint object will performed patch-based inpainting.
    Usage: create the object with parameters, call object.resolve().

    Parameters
    ----------
    image : array
        The image to inpaint.
    mask : array
        The mask of the same size as the image, all value > 0 will be inpainted.
    patch_size : int
        The size of one square patch.
    overlap_size : int
        The size of the overlap between patch.
    training_area : array
        The mask of the same size as the image, all value > 0 will be used for training.
    window_step : int
        The shape of the elementary n-dimensional orthotope of the rolling window view. If None will be autocomputed. Can lead to a RAM saturation if to small.
    mirror_hor : bool
        Compute the horizontal mirror of each patch for training.
    mirror_vert : bool
        Compute the vertical mirror of each patch for training.
    rotation : list
        Compute the given rotations in degrees of each patch for training.
    method : str
        Method to use for blending adjacent patches. blend: feathering blending ; linear: mean blending ; gaussian: gaussian blur blending ; None: no blending.

    """

    def __init__(
        self,
        image,
        mask,
        patch_size,
        overlap_size,
        shape,
        candidate=None,
        indexes=None,
        window_step=None,
        method="blend",
        model_path=None
    ):
        self.restriction_on_candidates = candidate
        self.max_val = np.max(image)
        self.dtype = image.dtype
        self.image = np.float32(image)
        self.image /= self.max_val
        self.original_shape = self.image.shape
        self.shape = shape
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.total_patches_count = 0
        self.example_patches = None
        self.mask = mask
        self.shift = 3
        self.blur_value = 5
        self.image_shape = (20, 1)
        self.model_path=model_path
        if self.mask is None:
            self.mask = np.ones_like(self.image)
        if window_step is None:
            self.window_step = np.max(np.shape(self.image)) // 30
        else:
            self.window_step = window_step
        # if rect is None:
        self.rects, indexes = self.compute_rect()
        # else:
        #     self.rects = [rect]
        self.method = method

        if indexes is not None:
            self.mask_indexes = indexes

        self.example_patches = self.compute_patches()

        self.kdtree = self.init_KDtrees()

        self.PARM_truncation = 0
        self.PARM_attenuation = 2

        #self.image = np.zeros_like(self.image)

        # if self.method == "blend":
        #     self.blending_mask = np.ones(
        #         (
        #             self.patch_size + 2 * self.overlap_size,
        #             self.patch_size + 2 * self.overlap_size,
        #             3,
        #         )
        #     )
        #     self.blending_mask[0 : self.overlap_size // 3, :, :] = 0
        #     self.blending_mask[:, 0 : self.overlap_size // 3, :] = 0
        #     self.blending_mask[-self.overlap_size // 3 : :, :, :] = 0
        #     self.blending_mask[:, -self.overlap_size // 3 : :, :] = 0
        #     self.blending_mask = gaussian(
        #         self.blending_mask,
        #         sigma=self.overlap_size // 2,
        #         preserve_range=True,
        #         channel_axis=2,
        #     )
        #     self.blending_mask = exposure.rescale_intensity(self.blending_mask)

    def get_areas(self, kernal_size):
        areas = []

        line_up, line_down, word_left, word_right = self.mask_indexes

        y_top_start = 0
        y_bottom_start = self.image.shape[0]
        x_left_start = 0
        x_right_start = self.image.shape[1]

        if kernal_size <= line_up:
            areas.append(self.image[y_top_start:line_up, :, :])
            y_top_start = line_up
        if kernal_size <= line_down:
            areas.append(self.image[y_bottom_start - line_down : y_bottom_start, :, :])
            y_bottom_start = y_bottom_start - line_down
        if kernal_size <= word_left:
            areas.append(
                self.image[y_top_start:y_bottom_start, x_left_start:word_left, :]
            )
        if kernal_size <= word_right:
            areas.append(
                self.image[
                    y_top_start:y_bottom_start,
                    x_right_start - word_right : x_right_start,
                    :,
                ]
            )

        return areas

    def view_as_windows_areas(self, kernel_size):
        result = []

        areas = self.get_areas(kernel_size)

        for area in areas:
            windows = view_as_windows(area, [kernel_size, kernel_size, 3], self.window_step)
            shape = np.shape(windows)
            windows = windows.reshape(shape[0] * shape[1], kernel_size, kernel_size, 3)
            result.append(windows)

        result = np.concatenate(result, axis=0)
        return result

    def first_nonzero(self, axis, invalid_val=-1):
        return np.where(
            self.mask.any(axis=axis), self.mask.argmax(axis=axis), invalid_val
        )

    def compute_rect(self):
        indexes = np.nonzero(self.mask)
        y_shift = indexes[0][0]
        x_shift = indexes[1][0]
        y_width = indexes[0][-1] - y_shift
        x_width = indexes[1][-1] - x_shift
        rect = [
            [
                x_shift,
                y_shift,
                x_width,
                y_width,
            ]
        ]
        test = np.zeros_like(self.image)
        
        test[y_shift : y_shift + y_width, x_shift : x_shift + x_width] = 255

        up = y_shift
        down = self.mask.shape[0] - (y_shift + y_width)
        left = x_shift
        right = self.mask.shape[1] - (x_shift + x_width)
        mask_indexes = (up, down, left, right)

        return rect, mask_indexes

    def compute_patches(self):
        kernel_size = self.patch_size
        #self.image[self.mask > 0] = np.nan

        result = view_as_windows(self.image, [kernel_size, kernel_size], self.window_step)
        shape = np.shape(result)
        result = result.reshape(shape[0] * shape[1], kernel_size, kernel_size)
        #result = self.view_as_windows_areas(kernel_size)

        self.total_patches_count = result.shape[0]

        return result

    def generate_maps(self, map_size=64):
        image_shape=self.image_shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_netG = torch.load(self.model_path, map_location=torch.device(device))
        model_netG.eval()

        h = image_shape[0] * map_size
        w = image_shape[1] * map_size

        result = np.zeros((h, w * 4), dtype=np.uint8)

        for i in range(0, image_shape[0]):
            for j in range(0, image_shape[1] * 4, 4):
                gi = self.generate_images(model_netG, "cpu")
                for k in range(4):
                    result[i * map_size:(i + 1) * map_size, (j + k) * map_size:((j + k) + 1) * map_size] = gi
                    gi = np.rot90(gi, -1)

        return result

    def get_combined_overlap(self, overlaps):
        shape = np.shape(overlaps[0])
        if len(shape) > 1:
            combined = np.zeros((shape[0], shape[1] * len(overlaps)))
            for i, j in enumerate(overlaps):
                combined[0 : shape[0], shape[1] * i : shape[1] * (i + 1)] = j
        else:
            combined = np.zeros((shape[0] * len(overlaps)))
            for i, j in enumerate(overlaps):
                combined[shape[0] * i : shape[0] * (i + 1)] = j
        return combined

    def init_KDtrees(self, leaf_size=25):
        top_overlap = self.example_patches[:, 0 : self.overlap_size, :]
        # bottom_overlap = self.example_patches[:, -self.overlap_size : :, :]
        left_overlap = self.example_patches[:, :, 0 : self.overlap_size]
        # right_overlap = self.example_patches[:, :, -self.overlap_size : :]

        shape_top = np.shape(top_overlap)
        # shape_bottom = np.shape(bottom_overlap)
        shape_left = np.shape(left_overlap)
        # shape_right = np.shape(right_overlap)

        flatten_top = top_overlap.reshape(shape_top[0], -1)
        # flatten_bottom = bottom_overlap.reshape(shape_bottom[0], -1)
        flatten_left = left_overlap.reshape(shape_left[0], -1)
        # flatten_right = right_overlap.reshape(shape_right[0], -1)

        # flatten_combined_4 = self.get_combined_overlap(
        #     [flatten_top, flatten_bottom, flatten_left, flatten_right]
        # )
        # flatten_combined_3 = self.get_combined_overlap(
        #     [flatten_top, flatten_left, flatten_right]
        # )
        # flatten_combined_3_bis = self.get_combined_overlap(
        #     [flatten_top, flatten_bottom, flatten_left]
        # )
        flatten_combined_2 = self.get_combined_overlap([flatten_top, flatten_left])
        # flatten_combined_2_bis = self.get_combined_overlap(
        #     [flatten_top, flatten_bottom]
        # )
        # flatten_combined_2_bis_1 = self.get_combined_overlap(
        #     [flatten_left, flatten_right]
        # )
        # flatten_combined_ltr = self.get_combined_overlap(
        #     [flatten_left, flatten_bottom, flatten_right]
        # )
        # flatten_combined_lb = self.get_combined_overlap([flatten_left, flatten_bottom])

        tree_top = KDTree(flatten_top, leafsize=leaf_size)
        tree_left = KDTree(flatten_left, leafsize=leaf_size)
        # tree_combined_4 = KDTree(flatten_combined_4, leafsize=leaf_size)
        # tree_combined_3 = KDTree(flatten_combined_3, leafsize=leaf_size)
        # tree_combined_3_bis = KDTree(flatten_combined_3_bis, leafsize=leaf_size)
        tree_combined_2 = KDTree(flatten_combined_2, leafsize=leaf_size)
        # tree_combined_2_bis = KDTree(flatten_combined_2_bis, leafsize=leaf_size)
        # tree_combined_2_bis_1 = KDTree(flatten_combined_2_bis_1, leafsize=leaf_size)
        # tree_combined_blr = KDTree(flatten_combined_ltr, leafsize=leaf_size)
        # tree_combined_lb = KDTree(flatten_combined_lb, leafsize=leaf_size)

        return {
            "t": tree_top,
            # "blr": tree_combined_blr,
            # "lb": tree_combined_lb,
            "l": tree_left,
            # "tblr": tree_combined_4,
            # "tlr": tree_combined_3,
            "tl": tree_combined_2
            # "tbl": tree_combined_3_bis,
            # "tb": tree_combined_2_bis,
            # "lr": tree_combined_2_bis_1,
        }

    def find_most_similar_patches(
        self, overlap_top, overlap_bottom, overlap_left, overlap_right, k=5
    ):
        # if (
        #     (overlap_top is not None)
        #     and (overlap_bottom is not None)
        #     and (overlap_left is not None)
        #     and (overlap_right is not None)
        # ):
        #     combined = self.get_combined_overlap(
        #         [
        #             overlap_top.reshape(-1),
        #             overlap_bottom.reshape(-1),
        #             overlap_left.reshape(-1),
        #             overlap_right.reshape(-1),
        #         ]
        #     )
        #     dist, ind = self.kdtree["tblr"].query([combined], k=k)
        # elif (
        #     (overlap_top is not None)
        #     and (overlap_bottom is None)
        #     and (overlap_left is not None)
        #     and (overlap_right is not None)
        # ):
        #     combined = self.get_combined_overlap(
        #         [
        #             overlap_top.reshape(-1),
        #             overlap_left.reshape(-1),
        #             overlap_right.reshape(-1),
        #         ]
        #     )
        #     dist, ind = self.kdtree["tlr"].query([combined], k=k)
        # elif (
        #     (overlap_top is None)
        #     and (overlap_bottom is not None)
        #     and (overlap_left is not None)
        #     and (overlap_right is not None)
        # ):
        #     combined = self.get_combined_overlap(
        #         [
        #             overlap_left.reshape(-1),
        #             overlap_bottom.reshape(-1),
        #             overlap_right.reshape(-1),
        #         ]
        #     )
        #     dist, ind = self.kdtree["blr"].query([combined], k=k)
        # elif (
        #     (overlap_top is None)
        #     and (overlap_bottom is not None)
        #     and (overlap_left is not None)
        #     and (overlap_right is None)
        # ):
        #     combined = self.get_combined_overlap(
        #         [overlap_left.reshape(-1), overlap_bottom.reshape(-1)]
        #     )
        #     dist, ind = self.kdtree["lb"].query([combined], k=k)
        # elif (
        #     (overlap_top is not None)
        #     and (overlap_bottom is not None)
        #     and (overlap_left is not None)
        #     and (overlap_right is None)
        # ):
        #     combined = self.get_combined_overlap(
        #         [
        #             overlap_top.reshape(-1),
        #             overlap_bottom.reshape(-1),
        #             overlap_left.reshape(-1),
        #         ]
        #     )
        #     dist, ind = self.kdtree["tbl"].query([combined], k=k)
        if (
            (overlap_top is not None)
            and (overlap_bottom is None)
            and (overlap_left is not None)
            and (overlap_right is None)
        ):
            combined = self.get_combined_overlap(
                [overlap_top.reshape(-1), overlap_left.reshape(-1)]
            )
            dist, ind = self.kdtree["tl"].query([combined], k=k)
        elif (
            (overlap_top is not None)
            and (overlap_bottom is None)
            and (overlap_left is None)
            and (overlap_right is None)
        ):
            dist, ind = self.kdtree["t"].query([overlap_top.reshape(-1)], k=k)
        elif (
            (overlap_top is None)
            and (overlap_bottom is None)
            and (overlap_left is not None)
            and (overlap_right is None)
        ):
            dist, ind = self.kdtree["l"].query([overlap_left.reshape(-1)], k=k)
        # elif (
        #     (overlap_top is not None)
        #     and (overlap_bottom is not None)
        #     and (overlap_left is None)
        #     and (overlap_right is None)
        # ):
        #     combined = self.get_combined_overlap(
        #         [overlap_top.reshape(-1), overlap_bottom.reshape(-1)]
        #     )
        #     dist, ind = self.kdtree["tb"].query([combined], k=k)
        # elif (
        #     (overlap_top is None)
        #     and (overlap_bottom is None)
        #     and (overlap_left is not None)
        #     and (overlap_right is not None)
        # ):
        #     combined = self.get_combined_overlap(
        #         [overlap_left.reshape(-1), overlap_right.reshape(-1)]
        #     )
        #     dist, ind = self.kdtree["lr"].query([combined], k=k)
        elif (
            (overlap_top is None)
            and (overlap_bottom is None)
            and (overlap_left is None)
            and (overlap_right is None)
        ):
            dist, ind = [None], [0]
        else:
            raise Exception(
                "ERROR: no valid overlap area is passed to -findMostSimilarPatch-"
            )
        dist = dist[0]
        ind = ind[0]
        return dist, ind

    def minCutPatch(self, image, patch, overlap_left, overlap_top):
        patch = cv2.cvtColor(patch.copy(), cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        h, w = patch.shape[:2]
        horizontal_mask = None
        vertical_mask = None
        overlap = self.overlap_size
        minCut = np.zeros_like(patch, dtype=bool)

        if overlap_left is not None:
            patch_overlap = patch[:, :overlap]
            image_overlap = image[:, :overlap]
            left = patch_overlap - image_overlap
            leftL2 = np.sum(left**2, axis=2)
            for i, j in enumerate(self.minCutPath(leftL2)):
                minCut[i, :j] = True

            translation_matrix = np.float32([[1, 0, self.shift], [0, 1, 0]])
            horizontal = cv2.warpAffine(minCut.astype(np.uint8) * 255, translation_matrix, (w, h), borderValue=(255, 255, 255))
            horizontal_mask = cv2.cvtColor(horizontal - (minCut.astype(np.uint8) * 255), cv2.COLOR_RGB2GRAY)

        if overlap_top is not None:
            patch_overlap = patch[:overlap, :]
            image_overlap = image[:overlap, :]
            up = patch_overlap - image_overlap
            upL2 = np.sum(up**2, axis=2)
            for j, i in enumerate(self.minCutPath(upL2.T)):
                minCut[:i, j] = True

            translation_matrix = np.float32([[1, 0, 0], [0, 1, self.shift]])
            vertical = cv2.warpAffine(minCut.astype(np.uint8) * 255, translation_matrix, (w, h), borderValue=(255, 255, 255))
            vertical_mask = cv2.cvtColor(vertical - (minCut.astype(np.uint8) * 255), cv2.COLOR_RGB2GRAY)

        np.copyto(patch, image, where=minCut)
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        if (horizontal_mask is not None) and (vertical_mask is not None):
            diff_mask = (horizontal_mask > 0) + (vertical_mask > 0)
        elif horizontal_mask is not None:
            diff_mask = horizontal_mask
        elif vertical_mask is not None:
            diff_mask = vertical_mask

        patch[diff_mask>0] = cv2.medianBlur(patch, self.blur_value)[diff_mask>0]

        return patch
    
    def minCutPath(self, errors):
    # dijkstra's algorithm vertical
        pq = [(error, [i]) for i, error in enumerate(errors[0])]
        heapq.heapify(pq)

        h, w = errors.shape
        seen = set()

        while pq:
            error, path = heapq.heappop(pq)
            curDepth = len(path)
            curIndex = path[-1]

            if curDepth == h:
                return path

            for delta in -1, 0, 1:
                nextIndex = curIndex + delta

                if 0 <= nextIndex < w:
                    if (curDepth, nextIndex) not in seen:
                        cumError = error + errors[curDepth, nextIndex]
                        heapq.heappush(pq, (cumError, path + [nextIndex]))
                        seen.add((curDepth, nextIndex))

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
        fake_img[fake_img > 1.0] = 1.0
        fake_img = (fake_img * 255).astype(np.uint8)
        if ReturnGray:
            fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2GRAY)
        return fake_img

    def generate_images(self, model_netG, device='cpu', image_shape=(64, 64)):
        fixed_noise_test = torch.rand(1, 100, 1, 1, device=device)
        generated_images = self.get_preprocess_image(model_netG(fixed_noise_test), True, image_shape)
        return generated_images

    def resolve(self):
        """
        Start the inpainting.
        """

        x0 = 0 #int(rect[0] - self.overlap_size)
        y0 = 0 #int(rect[1] - self.overlap_size)

        h, w  = self.shape

        numPatchesHigh = math.ceil((h - self.patch_size + self.overlap_size) / (self.patch_size - self.overlap_size)) + 1 or 1
        numPatchesWide = math.ceil((w - self.patch_size + self.overlap_size) / (self.patch_size - self.overlap_size)) + 1 or 1

        h = (numPatchesHigh * self.patch_size) - (numPatchesHigh - 1) * self.overlap_size
        w = (numPatchesWide * self.patch_size) - (numPatchesWide - 1) * self.overlap_size

        result_image = np.zeros((h, w), dtype=np.float32)

        step_x = numPatchesWide
        step_y = numPatchesHigh

        for i in range(step_y):  # Y
            for j in range(step_x):  # X
                if (i == 0) and (j == 0):
                    candidate_indices = np.random.randint(
                        0, self.example_patches.shape[0], 1)
                    result_image[0:self.patch_size, 0:self.patch_size] = self.example_patches[candidate_indices]
                    x0 += self.patch_size

                    continue

                x = max(0, x0)
                y = max(0, y0)

                y_patch_shift = y + self.patch_size 

                if y0 > 0:
                    if x > 0:
                        overlap_top = result_image[
                            y : y + self.overlap_size, x-self.overlap_size:(x - self.overlap_size) + self.patch_size]
                    else:
                        overlap_top = result_image[
                            y : y + self.overlap_size, x:x + self.patch_size]
                else:
                    overlap_top = None

                if x0 > 0:
                    overlap_left = result_image[
                        y:y_patch_shift, x-self.overlap_size:x]
                else:
                    overlap_left = None

                dist, ind = self.find_most_similar_patches(
                    overlap_top, None, overlap_left, None
                )

                if (dist is not None) and (float("inf") not in dist):
                    # probabilities = self.distances2probability(
                    #     dist, self.PARM_truncation, self.PARM_attenuation
                    # )
                    patch_id = ind[0]#np.random.choice(ind, 1, p=probabilities)
                else:
                    patch_id = np.random.choice(1, self.total_patches_count)

                # self.image[y:y_double_shift, x:x_double_shift] = self.merge(
                #     self.image[y:y_double_shift, x:x_double_shift],
                #     self.example_patches[patch_id[0], :, :],
                #     method=self.method,
                # )

                if x0 == 0:
                    x_left_position = x0
                    x_right_position = x0 + self.patch_size
                else:
                    x_left_position = x0 - self.overlap_size
                    x_right_position = x0 + self.patch_size - self.overlap_size
                image = result_image[y:y_patch_shift, x_left_position:x_right_position]
                patch = self.example_patches[patch_id, :, :]
                result_image[y:y_patch_shift, x_left_position:x_right_position] = self.minCutPatch(
                    image,
                    patch,
                    overlap_left, overlap_top
                )
                if x0 == 0:
                    x0 += self.patch_size
                else:
                    x0 += self.patch_size - self.overlap_size

                self.init_new_patches()

            x0 = 0
            y0 += self.patch_size - self.overlap_size
        #result_image = (result_image * 255).astype(np.uint8)
        return result_image

    def init_new_patches(self):
        map = self.generate_maps()
        self.max_val = np.max(map)
        self.dtype = map.dtype
        self.image = np.float32(map)
        self.image /= self.max_val
        self.example_patches = self.compute_patches()
        self.kdtree = self.init_KDtrees()

    def merge(self, image_0, image_1, method="linear"):
        image_1 = image_1[0 : image_0.shape[0], 0 : image_0.shape[1]]
        non_zeros = ~np.isnan(image_0)  # Overlap area
        zeros = np.isnan(image_0)  # patch_size area
        if method == "linear":
            image_0[zeros] = image_1[zeros]
            image_0[non_zeros] = (image_0[non_zeros] + image_1[non_zeros]) / 2
        elif method == "gaussian":
            image_0 = image_1
            image_0[non_zeros] = gaussian(
                image_0[non_zeros], sigma=1, preserve_range=True, channel_axis=2
            )
        elif method == "blend":
            self.blending_mask = self.blending_mask[
                0 : image_0.shape[0], 0 : image_0.shape[1]
            ]
            image_0[zeros] = image_1[zeros] * self.blending_mask[zeros]
            image_0[non_zeros] = (
                image_0[non_zeros] * (1 - self.blending_mask[non_zeros])
                + image_1[non_zeros] * self.blending_mask[non_zeros]
            )
        elif method == "quilting":
             pass
        else:
            raise ValueError("Invalid method")
        return image_0

    def distances2probability(self, distances, PARM_truncation, PARM_attenuation):
        if np.unique(np.array(distances)).size == 1:
            probabilities = np.ones_like(distances) * 0.5
        else:
            probabilities = 1 - distances / np.max(distances)
        probabilities *= probabilities > PARM_truncation
        probabilities = pow(probabilities, PARM_attenuation)  # attenuate the values
        # check if we didn't truncate everything!
        if np.sum(probabilities) == 0:
            # then just revert it
            probabilities = 1 - distances / np.max(distances)
            probabilities *= probabilities > PARM_truncation * np.max(
                probabilities
            )  # truncate the values (we want top truncate%)
            probabilities = pow(probabilities, PARM_attenuation)
        probabilities /= np.sum(probabilities)  # normalize so they add up to one

        return probabilities



def main():
    pass


if __name__ == "__main__":
    main()