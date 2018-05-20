from glob import glob
from os import remove

from fire import Fire
from tqdm import tqdm
import cv2
from joblib import Parallel, delayed


def fix_thread_safe(x):
    with open(x, 'rb') as f:
        data = f.read()
        check_chars = data[-2:]
        if not check_chars == b'\xff\xd9':
            with open(x, 'wb') as f:
                f.write(data)

    img = cv2.imread(x)
    if img is None or img[-5:, :, :].std() == 0:
        remove(x)
        return


def main(path):
    files = glob(path)
    Parallel(n_jobs=-1)(delayed(fix_thread_safe)(x) for x in tqdm(files))


if __name__ == '__main__':
    Fire(main)
