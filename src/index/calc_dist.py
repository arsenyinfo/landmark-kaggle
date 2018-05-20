import numpy as np
from tqdm import tqdm
from torch.tensor import Tensor
from torch.nn import CosineSimilarity
from h5py import File
import pandas as pd
import joblib as jl
from glog import logger

from fire import Fire

SHAPE = 1024


def arr(data):
    data = Tensor(data.astype('float32'))
    return data


def cosine(a, b):
    a = a.cuda()
    c = CosineSimilarity(dim=1)(a, b).cpu().numpy().reshape(-1)
    c *= 255
    return c.astype('uint8')


def get_data(prefix, name):
    ids = pd.read_csv(f'data/{prefix}.csv')['id']
    f = File(f'data/features_{prefix}_{name}.h5', 'r')
    vectors = f['result'][:]
    f.close()

    return ids, vectors


def check_empty(x):
    return 1 if x.mean() == 0 and x.var() == 0 else 0


def calculate_distances(name):
    index_ids, index_vectors = get_data('index', name)
    test_ids, test_vectors = get_data('test', name)
    logger.info('data is read')

    index_vectors, test_vectors = map(arr, (index_vectors, test_vectors))
    logger.info('tensors are ready')

    index_ids = index_ids
    test_ids = test_ids

    shape = len(test_ids), len(index_ids)

    file = File('data/distances.h5', 'w')
    result = file.create_dataset('result', shape=shape, dtype=np.uint8)
    logger.info('h5 file is ready')

    index_vectors = index_vectors.view(-1, SHAPE).cuda()
    for i in tqdm(np.arange(shape[0]), desc='calculating cosine'):
        c = cosine(test_vectors[i].view(-1, SHAPE), index_vectors)
        result[i, :] = c

    for i, v in tqdm(zip(index_ids, index_vectors), desc='removing empty pics'):
        if v is None:
            result[:, i] = 255

    file.close()


def process(i, id_, is_empty, index_ids, dataset):
    try:
        return _process(i, id_, is_empty, index_ids, dataset)
    except Exception:
        logger.exception(f'Can not process {i} {id_}')
        return {'id': id_, 'images': ''}


def _process(i, id_, is_empty, index_ids, dataset):
    if is_empty:
        return {'id': id_, 'images': ' '}
    sim = dataset[i, :][:]
    idx = np.argsort(sim)[-100:]

    return {'id': id_, 'images': ' '.join(map(str, index_ids[idx]))}


def make_submit(name):
    index_ids, index_vectors = get_data('index', name)
    test_ids, test_vectors = get_data('test', name)
    test_vector_empty = jl.Parallel(n_jobs=-1, backend='threading')(jl.delayed(check_empty)(x) for x in test_vectors)

    file = File('data/distances.h5', 'r')
    dataset = file['result']

    result = jl.Parallel(n_jobs=-1, backend='threading')(jl.delayed(process)(i, id_, empty, index_ids, dataset)
                                                         for i, (id_, empty) in
                                                         tqdm(enumerate(zip(test_ids, test_vector_empty)), desc='test',
                                                              total=len(test_ids)))

    file.close()
    result = pd.DataFrame(result)
    result.to_csv('result/retrieval.csv', index=False)


if __name__ == '__main__':
    Fire(calculate_distances)
    Fire(make_submit)
