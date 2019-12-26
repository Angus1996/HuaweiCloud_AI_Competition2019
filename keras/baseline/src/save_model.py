# -*- coding: utf-8 -*-
import os
import moxing as mox  # 华为自研模块moxing，本地机器无法安装，仅可在华为云ModelArts平台上使用，
# moxing文档请查看 https://github.com/huaweicloud/ModelArts-Lab/tree/master/docs/moxing_api_doc
import tensorflow as tf
from keras import backend
from keras.optimizers import adam

from train import model_fn


def load_weights(model, weighs_file_path):
    if mox.file.exists(weighs_file_path) and (not mox.file.is_directory(weighs_file_path)):
        print('load weights from %s' % weighs_file_path)
        if weighs_file_path.startswith('s3://'):
            weighs_file_name = weighs_file_path.rsplit('/', 1)[1]
            mox.file.copy(weighs_file_path, '/cache/tmp/' + weighs_file_name)
            weighs_file_path = '/cache/tmp/' + weighs_file_name
            model.load_weights(weighs_file_path)
            os.remove(weighs_file_path)
        else:
            model.load_weights(weighs_file_path)
        print('load weights success')
    else:
        print('load weights failed! Please check weighs_file_path')
    return model


def save_pb_model(FLAGS, model):
    if FLAGS.mode == 'train':
        pb_save_dir_local = FLAGS.train_local
        pb_save_dir_obs = FLAGS.train_url
    elif FLAGS.mode == 'save_pb':
        freeze_weights_file_dir = FLAGS.freeze_weights_file_path.rsplit('/', 1)[0]
        if freeze_weights_file_dir.startswith('s3://'):
            pb_save_dir_local = '/cache/tmp'
            pb_save_dir_obs = freeze_weights_file_dir
        else:
            pb_save_dir_local = freeze_weights_file_dir
            pb_save_dir_obs = pb_save_dir_local

    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input_img': model.input}, outputs={'output_score': model.output})
    builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(pb_save_dir_local, 'model'))
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=backend.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images': signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()
    print('save pb to local path success')

    if pb_save_dir_obs.startswith('s3://'):
        mox.file.copy_parallel(os.path.join(pb_save_dir_local, 'model'),
                               os.path.join(pb_save_dir_obs, 'model'))
        print('copy pb to %s success' % pb_save_dir_obs)

    mox.file.copy(os.path.join(FLAGS.deploy_script_path, 'config.json'),
                  os.path.join(pb_save_dir_obs, 'model/config.json'))
    mox.file.copy(os.path.join(FLAGS.deploy_script_path, 'customize_service.py'),
                  os.path.join(pb_save_dir_obs, 'model/customize_service.py'))
    if mox.file.exists(os.path.join(pb_save_dir_obs, 'model/config.json')) and \
            mox.file.exists(os.path.join(pb_save_dir_obs, 'model/customize_service.py')):
        print('copy config.json and customize_service.py success')
    else:
        print('copy config.json and customize_service.py failed')
    return pb_save_dir_local


def load_weights_save_pb(FLAGS):
    optimizer = adam(lr=FLAGS.learning_rate, clipnorm=0.001)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = model_fn(FLAGS, objective, optimizer, metrics)
    model = load_weights(model, FLAGS.freeze_weights_file_path)
    save_pb_model(FLAGS, model)
