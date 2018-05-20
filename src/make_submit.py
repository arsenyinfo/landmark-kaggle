from collections import defaultdict
from functools import partial

from keras.models import load_model
from keras.applications.xception import preprocess_input
from keras.applications.mobilenet import relu6, DepthwiseConv2D
import pandas as pd
from tqdm import tqdm
import numpy as np
from fire import Fire

from src.utils import logger, read_image, ten_crops, resize


def _process_file(path, target_shape):
    img = read_image(path, prefix='test')
    if img is None:
        return
    img = resize(img, target_shape)
    return ten_crops(img, target_shape)


def main(model_path, target_shape=224, batch_size=8, agg='mean'):
    model = load_model(model_path,
                       custom_objects={'DepthwiseConv2D': DepthwiseConv2D,
                                       'relu6': relu6},
                       compile=False)

    df = pd.read_csv('data/test.csv')
    ids = df.id
    logger.info('Dataframe has been read')

    result = []

    n_batches = ids.shape[0] // batch_size
    process_file = partial(_process_file, target_shape=target_shape)

    for batch_ids in tqdm(np.array_split(ids, n_batches), desc='batches processed'):
        imgs, valid_ids = [], []

        for id_ in batch_ids:
            crops = process_file(id_)
            if crops is None:
                result.append({'id': id_,
                               'landmarks': ''})
            else:
                imgs += crops
                valid_ids += [id_] * 10

        batch_data = np.array(imgs)
        batch_data = preprocess_input(batch_data.astype('float32'))
        batch_pred = model.predict(batch_data)

        acc = defaultdict(lambda: np.zeros(batch_pred.shape[1]))
        for id_, pred in zip(valid_ids, batch_pred):
            if agg == 'mean':
                acc[id_] += pred
            elif agg == 'max':
                acc[id_] = np.max([acc[id_], pred], axis=0)
            else:
                raise ValueError(f'aggregation {agg} is not supported')

        for id_ in acc:
            pred = acc[id_]
            idx = pred.argmax()

            result.append({'id': id_,
                           'landmarks': f'{idx} {pred[idx]}'})

    df = pd.DataFrame(result)
    df.to_csv('result/pred.csv', index=False)


if __name__ == '__main__':
    Fire(main)
