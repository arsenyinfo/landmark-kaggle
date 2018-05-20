import tracemalloc

tracemalloc.start()

import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.applications.inception_resnet_v2 import preprocess_input
from tqdm import tqdm
from h5py import File
from fire import Fire

from src.utils import read_image, resize, five_crops
from src.siamese.fit import l2_norm, pairwise_cosine_sim


def _process_file(path, target_shape, prefix):
    img = read_image(path, prefix=prefix)
    if img is None:
        return
    img = resize(img, int(target_shape * 1.5))
    return five_crops(img, target_shape)


def main(model_path, prefix='test', name='crops', batch_size=64, target_shape=224):
    files = pd.read_csv(f'data/{prefix}.csv')
    ids = files['id']

    base_model = load_model(model_path, custom_objects={'DepthwiseConv2D': DepthwiseConv2D,
                                                        'relu6': relu6,
                                                        'l2_norm': l2_norm,
                                                        'pairwise_cosine_sim': pairwise_cosine_sim})

    # model = Model(base_model.get_layer('model_1').inputs[0], base_model.get_layer('model_1').outputs[0])

    model = Model(base_model.input, base_model.get_layer('global_average_pooling2d_1').output)

    batch_ids, batch_images = [], []

    file = File(f'data/features_{prefix}_{name}.h5', 'w')
    result = file.create_dataset('result', shape=(len(ids), 1024), dtype=np.float16)
    result[:, :] = 0

    for i, id_ in tqdm(enumerate(ids), total=len(ids)):

        crops = _process_file(path=id_, target_shape=target_shape, prefix=prefix)
        if crops is None:
            continue

        for crop in crops:
            batch_ids.append(i)
            batch_images.append(crop)

            if len(batch_ids) == batch_size:
                features = model.predict(preprocess_input(np.array(batch_images).astype('float32')))
                for j, f in zip(batch_ids, features.astype('float16')):
                    result[j] += f
                batch_ids, batch_images = [], []

    if len(batch_images):
        features = model.predict(preprocess_input(np.array(batch_images).astype('float32')))
        for j, f in zip(batch_ids, features.astype('float16')):
            result[j] += f

    file.close()


if __name__ == '__main__':
    Fire(main)
