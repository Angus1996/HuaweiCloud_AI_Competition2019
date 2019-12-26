# -*- coding: utf-8 -*-
import os
import multiprocessing
from glob import glob
import moxing as mox  # 华为自研模块moxing，本地机器无法安装，仅可在华为云ModelArts平台上使用，
# moxing文档请查看 https://github.com/huaweicloud/ModelArts-Lab/tree/master/docs/moxing_api_doc
import numpy as np
from keras import backend
from keras.models import Model
from keras.optimizers import adam
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import TensorBoard, Callback, EarlyStopping
from keras import regularizers

from data_gen import data_flow
from models.resnet50 import ResNet50

backend.set_image_data_format('channels_last')


def model_fn(FLAGS, objective, optimizer, metrics):
    """
    pre-trained resnet50 model
    """
    base_model = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='sigmoid', kernel_regularizer=regularizers.l1(0.0001))(x)
    x = Dropout(rate=0.3)(x)
    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


class LossHistory(Callback):
    def __init__(self, FLAGS):
        super(LossHistory, self).__init__()
        self.FLAGS = FLAGS

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.FLAGS.snapshot_freq == 0:
            save_path = os.path.join(self.FLAGS.train_local, 'weights_%03d_%.4f.h5' % (epoch, logs.get('val_acc')))
            self.model.save_weights(save_path)
            if self.FLAGS.train_url.startswith('s3://'):
                save_url = os.path.join(self.FLAGS.train_url, 'weights_%03d_%.4f.h5' % (epoch, logs.get('val_acc')))
                mox.file.copy(save_path, save_url)
            print('save weights file', save_path)

        if self.FLAGS.keep_weights_file_num > -1:
            weights_files = glob(os.path.join(self.FLAGS.train_local, '*.h5'))
            if len(weights_files) >= self.FLAGS.keep_weights_file_num:
                weights_files.sort(key=lambda file_name: os.stat(file_name).st_ctime, reverse=True)
                for file_path in weights_files[self.FLAGS.keep_weights_file_num:]:
                    os.remove(file_path)  # only remove weights files on local path


def train_model(FLAGS):
    # data flow generator
    train_sequence, validation_sequence = data_flow(FLAGS.data_local, FLAGS.batch_size,
                                                    FLAGS.num_classes, FLAGS.input_size)

    optimizer = adam(lr=FLAGS.learning_rate, decay=1e-6,clipnorm=0.001)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = model_fn(FLAGS, objective, optimizer, metrics)
    if FLAGS.restore_model_path != '' and mox.file.exists(FLAGS.restore_model_path):
        if FLAGS.restore_model_path.startswith('s3://'):
            restore_model_name = FLAGS.restore_model_path.rsplit('/', 1)[1]
            mox.file.copy(FLAGS.restore_model_path, '/cache/tmp/' + restore_model_name)
            model.load_weights('/cache/tmp/' + restore_model_name)
            os.remove('/cache/tmp/' + restore_model_name)
        else:
            model.load_weights(FLAGS.restore_model_path)
        print('restore parameters from %s success' % FLAGS.restore_model_path)

    if not os.path.exists(FLAGS.train_local):
        os.makedirs(FLAGS.train_local)
    tensorboard = TensorBoard(log_dir=FLAGS.train_local, batch_size=FLAGS.batch_size)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=2)
    history = LossHistory(FLAGS)
    model.fit_generator(
        train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=FLAGS.max_epochs,
        verbose=1,
        callbacks=[history, tensorboard, early_stopping],
        validation_data=validation_sequence,
        max_queue_size=10,
        workers=int(multiprocessing.cpu_count() * 0.7),
        use_multiprocessing=True,
        shuffle=True
    )

    print('training done!')

    # 将训练日志拷贝到OBS，然后可以用 ModelArts 训练作业自带的tensorboard查看训练情况
    if FLAGS.train_url.startswith('s3://'):
        files = mox.file.list_directory(FLAGS.train_local)
        for file_name in files:
            if file_name.startswith('enevts'):
                mox.file.copy(os.path.join(FLAGS.train_local, file_name), os.path.join(FLAGS.train_url, file_name))
        print('save events log file to OBS path: ', FLAGS.train_url)

    pb_save_dir_local = ''
    if FLAGS.deploy_script_path != '':
        from save_model import save_pb_model
        # 默认将最新的模型保存为pb模型，您可以使用python run.py --mode=save_pb ... 将指定的h5模型转为pb模型
        pb_save_dir_local = save_pb_model(FLAGS, model)

    if FLAGS.deploy_script_path != '' and FLAGS.test_data_url != '':
        print('test dataset predicting...')
        from inference import infer_on_dataset
        accuracy, result_file_path = infer_on_dataset(FLAGS.test_data_local, FLAGS.test_data_local, os.path.join(pb_save_dir_local, 'model'))
        if accuracy is not None:
            metric_file_name = os.path.join(FLAGS.train_url, 'metric.json')
            metric_file_content = '{"total_metric": {"total_metric_values": {"accuracy": %0.4f}}}' % accuracy
            with mox.file.File(metric_file_name, "w") as f:
                f.write(metric_file_content + '\n')
            if FLAGS.train_url.startswith('s3://'):
                result_file_path_obs = os.path.join(FLAGS.train_url, 'model', os.path.basename(result_file_path))
                mox.file.copy(result_file_path, result_file_path_obs)
                print('accuracy result file has been copied to %s' % result_file_path_obs)
        else:
            print('accuracy is None')
    print('end')
