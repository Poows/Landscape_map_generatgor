import numpy as np
import cv2
import sys
import heapq
import math
from skimage import io, util
import torchvision.utils as vutils
from Landscape_generator import LandscapeGenerator, Generator
import torch


def randomPatch(texture, patchLength):
    h, w, _ = texture.shape
    i = 0 #np.random.randint(h - patchLength)
    j = 0 #np.random.randint(w - patchLength)

    return texture[i:i+patchLength, j:j+patchLength]


def minCutPath(errors):
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


def minCutPatch(patch, patchLength, overlap, res, y, x):
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)
    horizontal_mask = None
    vertical_mask = None
    h, w = patch.shape[:2]

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = np.sum(left**2, axis=2)
        for i, j in enumerate(minCutPath(leftL2)):
            minCut[i, :j] = True

        translation_matrix = np.float32([[1, 0, 3], [0, 1, 0]])
        horizontal = cv2.warpAffine(minCut.astype(np.uint8) * 255, translation_matrix, (w, h), borderValue=(255, 255, 255))
        horizontal_mask = cv2.cvtColor(horizontal - (minCut.astype(np.uint8) * 255), cv2.COLOR_RGB2GRAY)

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath(upL2.T)):
            minCut[:i, j] = True

        translation_matrix = np.float32([[1, 0, 0], [0, 1, 3]])
        vertical = cv2.warpAffine(minCut.astype(np.uint8) * 255, translation_matrix, (w, h), borderValue=(255, 255, 255))
        vertical_mask = cv2.cvtColor(vertical - (minCut.astype(np.uint8) * 255), cv2.COLOR_RGB2GRAY)

    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)

    if (horizontal_mask is not None) and (vertical_mask is not None):
        diff_mask = (horizontal_mask > 0) + (vertical_mask > 0)
    elif horizontal_mask is not None:
        diff_mask = horizontal_mask
    elif vertical_mask is not None:
        diff_mask = vertical_mask

    # h, w = patch.shape[:2]
    # translation_matrix = np.float32([[1, 0, 3], [0, 1, 0]])
    # dst = cv2.warpAffine(minCut.astype(np.uint8) * 255, translation_matrix, (w, h), borderValue=(255, 255, 255))
    # diff_mask = dst - (minCut.astype(np.uint8) * 255)

    return patch, diff_mask


def L2OverlapDiff(patch, patchLength, overlap, res, y, x):
    error = 0

    if x > 0:
        left = patch[:, :overlap] - res[y:y+patchLength, x:x+overlap]
        error += np.sum(left**2)

    if y > 0:
        up   = patch[:overlap, :] - res[y:y+overlap, x:x+patchLength]
        error += np.sum(up**2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
        error -= np.sum(corner**2)

    return error


def randomBestPatch(texture, patchLength, overlap, res, y, x, used_patches):
    h, w, _ = texture.shape
    #errors = np.zeros((h - patchLength, w - patchLength))
    errors = np.full((h, w), np.inf)

    for i in range(0, h, patchLength):
        for j in range(0, w, patchLength):
            if (i, j) not in used_patches:
                patch = texture[i:i+patchLength, j:j+patchLength]
                e = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
                errors[i, j] = e

    i, j = np.unravel_index(np.argmin(errors), errors.shape)

    index_length = j // patchLength
    position_number = index_length % 4
    k = index_length - position_number
    #index_position = j - position_number * patchLength
    index_position = k * patchLength

    for delete_position in range(4):
        errors[i, (index_position + delete_position * patchLength)] = 0
        used_patches.append((i, (index_position + delete_position * patchLength)))
        #texture[i:(i + patchLength), (index_position + delete_position * patchLength):(index_position + (delete_position + 1) * patchLength)] = 0

    best_patch = texture[i:i+patchLength, j:j+patchLength]

    return best_patch


def quilt(texture, patchLength, shape, overlap=0.2, mode="cut", sequence=False, use_global_map=False):
    texture = util.img_as_float(texture)
    texture = cv2.cvtColor(texture.astype(np.float32), cv2.COLOR_GRAY2RGB)
    h, w = shape

    overlap = math.ceil(overlap * patchLength)

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    # numPatchesHigh, numPatchesWide = numPatches
    used_patches = []
    used_patches.append((0, 0))

    h = (numPatchesHigh * patchLength) - (numPatchesHigh - 1) * overlap
    w = (numPatchesWide * patchLength) - (numPatchesWide - 1) * overlap

    if len(texture.shape) != 2:
        res = np.zeros((h, w, texture.shape[2]))
    else:
        res = np.zeros((h, w))

    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "random":
                patch = randomPatch(texture, patchLength)
            elif mode == "best":
                patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
            elif mode == "cut":
                patch = randomBestPatch(texture, patchLength, overlap, res, y, x, used_patches)
                patch, mask = minCutPatch(patch, patchLength, overlap, res, y, x)
                patch[mask>0] = cv2.blur(patch, (3, 3))[mask>0]
            
            res[y:y+patchLength, x:x+patchLength] = patch

            if sequence:
                io.imshow(res)
                io.show()

    if use_global_map:
        h, w = res.shape[:2]
        global_map = cv2.resize(generate_maps((1, 1)), (w, h), interpolation=cv2.INTER_CUBIC)
        global_map = util.img_as_float(global_map)
        global_map[global_map>0.7] = 1
        global_map[global_map<=0.7] /= 0.7
        res = util.img_as_float(res)
        res = res * global_map
    return res


def get_preprocess_image(TensorImage, ReturnGray, shape=(64, 64)):
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


def generate_images(model_netG, device, image_shape=(64, 64)):
    fixed_noise_test = torch.rand(1, 100, 1, 1, device=device)
    generated_images = get_preprocess_image(model_netG(fixed_noise_test), True, image_shape)
    return generated_images


def generate_maps(image_shape=(20, 20), map_size=64):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_netG = torch.load("new_nudes/DCGAN_netG.pt", map_location=torch.device(device))
    model_netG.eval()

    h = image_shape[0] * map_size
    w = image_shape[1] * map_size

    result = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            gi = cv2.cvtColor(generate_images(model_netG, "cpu"), cv2.COLOR_GRAY2RGB)
            result[i * map_size:(i + 1) * map_size, j * map_size:(j + 1) * map_size] = gi

    return result


def main():
    texture = generate_maps()#cv2.imread("result_map.png")
    q = quilt(texture, 64, (5, 5), mode="cut")
    q = (q * 255).astype(np.uint8)
    cv2.imwrite("synthesys_map2.png", q)


if __name__ == "__main__":
    sys.exit(main())