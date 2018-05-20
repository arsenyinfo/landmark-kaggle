import numpy as np


def gamma_correction(x, gamma=None):
    if gamma is None:
        gamma = np.random.randint(80, 121) / 100.
    x = x.astype('float32') / 255.
    x = np.power(x, gamma)
    return x * 255


def augment(x):
    augs = (np.fliplr,
            gamma_correction,
            None)
    f = np.random.choice(augs)

    if f is not None:
        return f(x)
    return x
