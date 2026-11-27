"""
画像の読み書き。

ilvr-for-anime-faceparts (87152b3) からの書き写し:
  center_crop_arr                                   <- guided_diffusion/image_datasets.py 126-143
                                                       (image_datasets.py は mpi4py と blobfile を import するので、この関数だけを写した)
  list_images, load_image, to_uint8_hwc, save_png   <- guided_diffusion/ilvr_eye.py 243-293
                                                       (list_images は {side} を含むディレクトリの分岐を除いた)
  str2bool                                          <- guided_diffusion/script_util.py 441-452
"""

import argparse
import os

import numpy as np
import torch as th
from PIL import Image


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def list_images(data_dir):
    """data_dir の画像名の一覧 (名前順)。"""
    return sorted(f for f in os.listdir(data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))


def load_image(path, image_size):
    """学習時の ImageDataset と同じ前処理 (center_crop_arr, [-1, 1]、反転なし)。[3 x H x W] を返す。"""
    with open(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    arr = center_crop_arr(pil_image.convert("RGB"), image_size).astype(np.float32) / 127.5 - 1
    return th.from_numpy(np.transpose(arr, [2, 0, 1]))


def to_uint8_hwc(x):
    """[3 x H x W] in [-1, 1] -> uint8 [H x W x 3]"""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(th.uint8).permute(1, 2, 0).cpu().numpy()


def save_png(x, path):
    Image.fromarray(to_uint8_hwc(x)).save(path)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
