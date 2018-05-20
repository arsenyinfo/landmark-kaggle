from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import resize, read_image, logger, five_crops
from src.aug import augment


class Dataset:
    def __init__(self, n_fold, batch_size, size=384, transform=None, train=True, file_list=None, aug=augment):
        self.transform = transform
        self.batch_size = batch_size
        self.size = size
        self.file_list = file_list
        self.aug = aug

        data, self.classes = self.get_data()
        val_data = data.pop(n_fold)
        if train:
            self.data = reduce(lambda x, y: x + y, data.values())
        else:
            self.data = val_data
        self.data = pd.DataFrame(self.data)

    def get_data(self):
        df = pd.read_csv(self.file_list)
        if 'landmark_id' in df.columns:
            df = df.sort_values('landmark_id')

        df = df.reset_index()
        n_classes = df['landmark_id'].max() + 1

        assert df['landmark_id'].unique().shape[0] == n_classes

        acc = defaultdict(list)
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            fold = i % 5
            acc[fold].append({'id': row['id'], 'landmark_id': row['landmark_id']})

        return acc, n_classes

    def _get_random_img(self):
        idx = np.random.randint(0, self.data.shape[0])
        row = self.data.iloc[idx]
        return row['id'], row['landmark_id'], None

    def _get_same_img(self, id_, landmark_id):
        subset = self.data[(self.data['landmark_id'] == landmark_id) & (self.data['id'] != id_)]
        l = subset.shape[0]
        if not l:
            logger.info(f'There is no same images as {landmark_id}')
            return self._get_other_img(id_, landmark_id)

        idx = np.random.randint(0, l)
        row = subset.iloc[idx]
        return row['id'], row['landmark_id'], 1

    def _get_other_img(self, id_, landmark_id):
        subset = self.data[(self.data['landmark_id'] != landmark_id) & (self.data['id'] != id_)]
        idx = np.random.randint(0, subset.shape[0])
        row = subset.iloc[idx]
        return row['id'], row['landmark_id'], 0

    def get_image(self, mode, **kwargs):
        f = {'random': self._get_random_img,
             'other': self._get_other_img,
             'same': self._get_same_img}[mode]

        while True:
            x, y, label = f(**kwargs)
            img = read_image(x)
            if img is None:
                continue
            crops = self.process_img(img)
            return crops[np.random.randint(0, 5)], x, y, label

    def process_img(self, img):
        img = resize(img, base=int(self.size * 1.5))
        return five_crops(img, size=self.size)

    def __next__(self):
        left, right, y_data = [], [], []
        while len(y_data) < self.batch_size:

            img, x, y, _ = self.get_image('random')
            left.append(img)

            label = np.random.choice((0, 0, 0, 0, 1))

            mode = 'same' if label else 'other'
            img, *_, label = self.get_image(mode, id_=x, landmark_id=y)

            right.append(img)
            y_data.append(label)

        left, right, y_data = map(lambda x: np.array(x).astype('float32'), (left, right, y_data))
        if self.transform:
            left, right = map(self.transform, (left, right))

        return [left, right], y_data
