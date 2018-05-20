from fire import Fire
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Activation, Lambda
from keras import backend as K
from keras.models import load_model, Model
from keras.applications.mobilenet import relu6, DepthwiseConv2D, preprocess_input

from src.siamese.dataset import Dataset
from src.aug import augment
from src.utils import logger


def get_callbacks(fold):
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.1, patience=5)
    checkpoint = ModelCheckpoint(f'models/siamese_{fold}_5f.h5',
                                 monitor='val_loss',
                                 save_best_only=True, verbose=0)
    callbacks = [es, reducer, checkpoint]
    return callbacks


def l2_norm(x, axis=None):
    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm


def pairwise_cosine_sim(A_B):
    a_tensor, b_tensor = A_B
    num = K.batch_dot(a_tensor, b_tensor, axes=1)
    den = l2_norm(a_tensor, axis=1) * l2_norm(b_tensor, axis=1)
    dist_mat = num / den

    return dist_mat


def build_siam(input_shape, feature_extractor):
    logger.info('Building new siamese model')

    inp1 = Input((input_shape, input_shape, 3))
    inp2 = Input((input_shape, input_shape, 3))

    f1 = feature_extractor(inp1)
    f2 = feature_extractor(inp2)

    distance = Lambda(pairwise_cosine_sim)([f1, f2])
    out = Activation('linear')(distance)

    return Model([inp1, inp2, ], out)


def fit_model(model_path, is_siam=False, batch_size=8, n_fold=1, shape=224, steps=100):
    n_fold = int(n_fold)
    batch_size = int(batch_size)
    preprocess = preprocess_input

    model = load_model(model_path,
                       custom_objects={'DepthwiseConv2D': DepthwiseConv2D,
                                       'relu6': relu6,
                                       'pairwise_cosine_sim': pairwise_cosine_sim,
                                       'l2_norm': l2_norm,
                                       },
                       compile=False)
    if not is_siam:
        feature_extractor = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d_1').output)
        model = build_siam(shape, feature_extractor)
    loss = 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    train = Dataset(n_fold=n_fold,
                    batch_size=batch_size,
                    transform=preprocess,
                    train=True,
                    size=shape,
                    aug=augment,
                    file_list='data/train.csv')

    val = Dataset(n_fold=n_fold,
                  batch_size=batch_size,
                  transform=preprocess,
                  train=False,
                  size=shape,
                  aug=augment,
                  file_list='data/train.csv')

    model.fit_generator(train,
                        epochs=100,
                        steps_per_epoch=steps,
                        validation_data=val,
                        workers=8,
                        validation_steps=steps // 10,
                        use_multiprocessing=False,
                        callbacks=get_callbacks(n_fold),
                        max_queue_size=50,
                        )


if __name__ == '__main__':
    Fire(fit_model)
