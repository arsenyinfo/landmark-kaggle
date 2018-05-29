from collections import defaultdict
from functools import reduce

from keras.utils import to_categorical
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import resize, crop, read_image
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

    def get_data(self):
        df = pd.read_csv(self.file_list)
        if 'landmark_id' in df.columns:
            df = df.sort_values('landmark_id')

        df = df.reset_index()
        n_classes = df['landmark_id'].max() + 1

        assert df['landmark_id'].unique().shape[0] == n_classes

        acc = defaultdict(list)
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            fold = i % 10
            acc[fold].append((row['id'], row['landmark_id']))

        return acc, n_classes

    def __next__(self):
        x_data, y_data = [], []
        while len(x_data) < self.batch_size:
            idx = np.random.randint(0, len(self.data))
            x, y = self.data[idx]
            x = read_image(x)

            if x is None:
                self.data.pop(idx)
                continue

            x = crop(resize(x, base=self.size), target_shape=self.size)
            x = self.aug(x)

            x_data.append(x)
            y_data.append(y)

        x_data, y_data = np.array(x_data).astype('float32'), np.array(y_data)
        if self.transform:
            x_data = self.transform(x_data)
        y_data = to_categorical(y_data, self.classes)
        return x_data, y_data
