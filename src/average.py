from glob import glob
from os import environ
from functools import reduce

from fire import Fire
from tqdm import tqdm
import numpy as np

environ['CUDA_VISIBLE_DEVICES'] = ''  # noqa

from keras import backend as K

from src.utils import load_model


def average(weights):
    if not len(weights[0]):
        return weights[0]
    return [np.mean(x, axis=0) for x in zip(*weights)]


def get_weights(path):
    model = load_model(path)
    weights = [l.get_weights() for l in model.layers]
    K.clear_session()
    # if we have a lot of loaded models, it becomes really slow
    return weights


def main(result, *masks):
    models = reduce(lambda x, y: x + y, map(glob, masks))
    weights = [get_weights(x) for x in tqdm(models, desc='loading')]

    model = load_model(models[0])

    for i, _ in enumerate(tqdm(model.layers, desc='averaging')):
        w = [x[i] for x in weights]
        w = average(w)
        model.layers[i].set_weights(w)

    model.save(result)


if __name__ == '__main__':
    Fire(main)
