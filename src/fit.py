from fire import Fire

from src.dataset import Dataset
from src.model import get_model, get_callbacks, preprocess_input
from src.aug import augment
from src.utils import load_model
from src.utils import logger


def fit_once(model, model_name, loss, train, val, n_fold, start_epoch, stage='init'):
    steps_per_epoch = len(train.data) // train.batch_size
    validation_steps = 100

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])
    history = model.fit_generator(train,
                                  epochs=500,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=val,
                                  workers=8,
                                  max_queue_size=32,
                                  use_multiprocessing=False,
                                  validation_steps=validation_steps,
                                  callbacks=get_callbacks(model_name, stage, n_fold),
                                  initial_epoch=start_epoch,
                                  )
    return model, max(history.epoch)


def fit_model(model_name, partial_fit=False, batch_size=32, n_fold=1, shape=384):
    n_fold = int(n_fold)
    batch_size = int(batch_size)

    if partial_fit:
        logger.info(f'Using existing model {model_name}')
        model = load_model(model_name)
        preprocess = preprocess_input
    else:
        model, preprocess = get_model(model_name, shape, n_classes=14951)

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

    frozen_epochs = 1
    steps_per_epoch = len(train.data) // batch_size // 100
    validation_steps = 100
    loss = 'categorical_crossentropy'

    if not partial_fit:
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        model.fit_generator(train,
                            epochs=frozen_epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val,
                            workers=8,
                            validation_steps=validation_steps,
                            use_multiprocessing=False,
                            callbacks=get_callbacks(model_name, 'frozen', n_fold),
                            max_queue_size=50,
                            )

        for layer in model.layers:
            layer.trainable = True
    else:
        model_name = model_name.split('/')[1].split('_')[0]
        logger.info(f'model name is {model_name}')

    fit_once(model=model,
             model_name=model_name,
             loss='categorical_crossentropy',
             train=train,
             val=val,
             start_epoch=frozen_epochs,
             n_fold=n_fold,
             )


if __name__ == '__main__':
    Fire(fit_model)
