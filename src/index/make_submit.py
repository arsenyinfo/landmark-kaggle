import numpy as np
import joblib as jl
from tqdm import tqdm
from h5py import File
import pandas as pd
from fire import Fire

from glog import logger


def process(i, v, id_, index_ids, dataset):
    try:
        return _process(i, v, id_, index_ids, dataset)
    except Exception:
        logger.exception(f'Can not process {i} {v} {id_}')
        return {'id': id_, 'images': ''}


def _process(i, v, id_, index_ids, dataset):
    if v is None:
        return {'id': id_, 'images': ''}
    else:
        sim = dataset[i, :][:]
        idx = np.argsort(sim)[-100:]
        print(sim[idx[0]])
        return {'id': id_, 'images': ' '.join(index_ids[idx])}


def main():
    index_ids, index_vectors = jl.load('data/index.bin')
    test_ids, test_vectors = jl.load('data/test.bin')
    file = File('data/distances.h5', 'r')
    dataset = file['result']

    result = jl.Parallel(n_jobs=-1, backend='threading')(jl.delayed(process)(i, v, id_, index_ids, dataset)
                                                         for i, (id_, v) in
                                                         tqdm(enumerate(zip(test_ids, test_vectors)), desc='test',
                                                              total=len(test_ids)))

    file.close()
    result = pd.DataFrame(result)
    result.to_csv('result/retrieval.csv', index=False)


if __name__ == '__main__':
    Fire(main)
