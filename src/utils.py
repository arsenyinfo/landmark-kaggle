import logging

import numpy as np
import cv2

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__name__)


def resize(img, base=384):
    w, h, _ = img.shape

    if w > h:
        return cv2.resize(img, (base, int(base * w / h)), cv2.INTER_CUBIC)
    else:
        return cv2.resize(img, (int(base * h / w), base), cv2.INTER_CUBIC)


def _get_delta(y):
    t = y / 2 + y * np.random.randn() / 2
    if 0 <= t <= y:
        return int(t)
    return _get_delta(y)


def crop(img, target_shape):
    a, b, _ = img.shape

    assert a == target_shape or b == target_shape

    if a != target_shape:
        y = a - target_shape
        delta = _get_delta(y)
        return img[delta: delta + target_shape, :, :]

    elif b != target_shape:
        y = b - target_shape
        delta = _get_delta(y)
        return img[:, delta: delta + target_shape, :]

    return img


def read_image(x, prefix='train'):
    try:
        img = cv2.imread(f'data/{prefix}/{x}.jpg')
    except Exception:
        logger.exception(f'Can not read {x}')
        return
    if img is not None:
        return img
    return


def five_crops(img, size):
    w, h, _ = img.shape

    w_crop = (w - size) // 2
    h_crop = (h - size) // 2

    return [
        img[:size, :size, :],
        img[:size, -size:, :],
        img[-size:, :size, :],
        img[-size:, -size:, :],
        img[w_crop: w_crop + size, h_crop: h_crop + size, :]
    ]


def ten_crops(img, size):
    crops = five_crops(img, size)
    flips = [np.fliplr(x) for x in crops]
    return crops + flips
