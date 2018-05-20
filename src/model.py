from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.nasnet import NASNetMobile
from keras.applications.densenet import DenseNet121
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout

from src.utils import logger


def get_callbacks(model_name, stage, fold):
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.1, patience=6)
    checkpoint = ModelCheckpoint(f'models/{model_name}_{stage}_{fold}.h5',
                                 monitor='val_loss',
                                 save_best_only=True, verbose=0)
    callbacks = [es, reducer, checkpoint]
    return callbacks


def get_prefit_model(model_name, n_classes):
    logger.info('Using model with weights')
    model = load_model(model_name)

    x = model.layers[-2].output
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)

    for layer in model.layers:
        layer.trainable = False
    return model, preprocess_input


def get_model(model_name, shape, n_classes):
    if model_name == 'xception':
        base_model = Xception(include_top=False, input_shape=(shape, shape, 3), pooling='avg')
        drop = .1
    elif model_name == 'incres':
        base_model = InceptionResNetV2(include_top=False, input_shape=(shape, shape, 3), pooling='avg')
        drop = .2
    elif model_name == 'inception':
        base_model = InceptionV3(include_top=False, input_shape=(shape, shape, 3), pooling='avg')
        drop = .1
    elif model_name == 'resnet':
        base_model = ResNet50(include_top=False, input_shape=(shape, shape, 3), pooling='avg')
        drop = .1
    elif model_name == 'mobilenet':
        base_model = MobileNet(include_top=False, input_shape=(shape, shape, 3), pooling='avg')
        drop = .1
    elif model_name == 'nasnet':
        base_model = NASNetMobile(include_top=False, input_shape=(shape, shape, 3), pooling='avg')
        drop = .1
    elif model_name == 'densenet':
        base_model = DenseNet121(include_top=False, input_shape=(shape, shape, 3), pooling='avg')
        drop = .1

    else:
        raise ValueError('Network name is unknown')

    x = base_model.output
    x = Dropout(drop)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model, preprocess_input
