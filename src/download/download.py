from urllib import request
from urllib.error import HTTPError
import os
from time import sleep
from sys import exit

import pandas as pd
from glog import logger
from tqdm import tqdm
from joblib import Parallel, delayed
from fire import Fire


def load_image(url, file_id, dir_name):
    i = 1
    while True:
        try:
            _load_image(url, file_id, dir_name)
            return
        except KeyboardInterrupt:
            exit()
        except HTTPError:
            return
        except Exception:
            logger.exception(f'{i} retries failed for {url}')
            i += 1
            if i > 10:
                return
            sleep(i)


def _load_image(url, file_id, dir_name):
    save_path = f'data/{dir_name}/{file_id}.jpg'
    if not os.path.exists(save_path):
        request.urlretrieve(url, save_path)


def main(filelist, output):
    df = pd.read_csv(filelist)
    Parallel(n_jobs=-1, backend='threading')(delayed(load_image)(row.get('url'), row.get('id'), output)
                                             for row in tqdm((row for _, row in df.iterrows()), total=df.shape[0]))


if __name__ == '__main__':
    Fire(main)
