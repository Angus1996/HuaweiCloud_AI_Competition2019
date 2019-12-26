# -*- coding: utf-8 -*-
"""
基于 resnet50 实现的图片分类代码
在 ModelArts Notebook 中的代码运行方法：
（1）训练
cd {run.py所在目录}
1）从零训练
python run.py --data_url='../datasets/train_data' --train_url='../model_snapshots' --num_classes=54 --deploy_script_path='./deploy_scripts' --test_data_url='../datasets/test_data' --max_epochs=6
2）加载已有模型继续训练
cd {run.py所在目录}
python run.py --data_url='../datasets/train_data' --train_url='../model_snapshots' --restore_model_path='../model_snapshots/weights_040_0.8480.h5' --num_classes=54 --deploy_script_path='./deploy_scripts' --test_data_url='../datasets/test_data' --max_epochs=6

（2）转pb
cd {run.py所在目录}
python run.py --mode=save_pb --deploy_script_path='./deploy_scripts' --freeze_weights_file_path='../model_snapshots/weights_000_0.9811.h5' --num_classes=54

（3）评价
cd {run.py所在目录}
1）评价单个h5文件
python run.py --mode=eval --eval_weights_path='../model_snapshots/weights_000_0.7020.h5' --num_classes=54 --test_data_url='../datasets/test_data'
2）评价批量h5文件
python run.py --mode=eval --eval_weights_path='../model_snapshots' --num_classes=54 --test_data_url='../datasets/test_data'
3）评价单个pb模型
python run.py --mode=eval --eval_pb_path='../model_snapshots/model' --num_classes=54 --test_data_url='../datasets/test_data'
"""
import os
import moxing as mox  # 华为自研模块moxing，本地机器无法安装，仅可在华为云ModelArts平台上使用，
# moxing文档请查看 https://github.com/huaweicloud/ModelArts-Lab/tree/master/docs/moxing_api_doc
import tensorflow as tf

tf.app.flags.DEFINE_string('mode', 'train', 'optional: train, save_pb, eval')
tf.app.flags.DEFINE_string('local_data_root', '/cache/',
                           'a directory used for transfer data between local path and OBS path')
# params for train
tf.app.flags.DEFINE_string('data_url', '', 'the training data path')
tf.app.flags.DEFINE_string('restore_model_path', '',
                           'a history model you have trained, you can load it and continue training')
tf.app.flags.DEFINE_string('train_url', '', 'the path to save training outputs')
tf.app.flags.DEFINE_integer('snapshot_freq', 5,
                            'every snapshot_freq steps will save a weights file')
tf.app.flags.DEFINE_integer('keep_weights_file_num', 20,
                            'the max num of weights files keeps, if set -1, means infinity')
tf.app.flags.DEFINE_integer('num_classes', 0, 'the num of classes which your task should classify')
tf.app.flags.DEFINE_integer('input_size', 224, 'the input image size of the model')
tf.app.flags.DEFINE_integer('batch_size', 128, '')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, '')
tf.app.flags.DEFINE_integer('max_epochs', 6, '')

# params for save pb
tf.app.flags.DEFINE_string('deploy_script_path', '',
                           'a path which contain config.json and customize_service.py, '
                           'if it is set, these two scripts will be copied to {train_url}/model directory')
tf.app.flags.DEFINE_string('freeze_weights_file_path', '',
                           'if it is set, the specified h5 weights file will be converted as a pb model, '
                           'only valid when {mode}=save_pb')

# params for evaluation
tf.app.flags.DEFINE_string('eval_weights_path', '', 'weights file path need to be evaluate')
tf.app.flags.DEFINE_string('eval_pb_path', '', 'a directory which contain pb file needed to be evaluate')
tf.app.flags.DEFINE_string('test_data_url', '', 'the test data path which contain image and label txt on obs')

tf.app.flags.DEFINE_string('data_local', '', 'the train data path on local')
tf.app.flags.DEFINE_string('train_local', '', 'the training output results on local')
tf.app.flags.DEFINE_string('test_data_local', '', 'the test data path on local')
tf.app.flags.DEFINE_string('tmp', '', 'a temporary path on local')

FLAGS = tf.app.flags.FLAGS


def check_args(FLAGS):
    if FLAGS.mode not in ['train', 'save_pb', 'eval']:
        raise Exception('FLAGS.mode error, should be train, save_pb or eval')
    if FLAGS.num_classes == 0:
        raise Exception('FLAGS.num_classes error, '
                        'should be a positive number associated with your classification task')

    if FLAGS.mode == 'train':
        if FLAGS.data_url == '':
            raise Exception('you must specify FLAGS.data_url')
        if not mox.file.exists(FLAGS.data_url):
            raise Exception('FLAGS.data_url: %s is not exist' % FLAGS.data_url)
        if FLAGS.restore_model_path != '' and (not mox.file.exists(FLAGS.restore_model_path)):
            raise Exception('FLAGS.restore_model_path: %s is not exist' % FLAGS.restore_model_path)
        if FLAGS.restore_model_path != '' and mox.file.is_directory(FLAGS.restore_model_path):
            raise Exception('FLAGS.restore_model_path must be a file path, not a directory, %s' % FLAGS.restore_model_path)
        if FLAGS.train_url == '':
            raise Exception('you must specify FLAGS.train_url')
        elif not mox.file.exists(FLAGS.train_url):
            mox.file.make_dirs(FLAGS.train_url)
        if FLAGS.deploy_script_path != '' and (not mox.file.exists(FLAGS.deploy_script_path)):
            raise Exception('FLAGS.deploy_script_path: %s is not exist' % FLAGS.deploy_script_path)
        if FLAGS.deploy_script_path != '' and mox.file.exists(FLAGS.train_url + '/model'):
            raise Exception(FLAGS.train_url +
                            '/model is already exist, only one model directoty is allowed to exist')
        if FLAGS.test_data_url != '' and (not mox.file.exists(FLAGS.test_data_url)):
            raise Exception('FLAGS.test_data_url: %s is not exist' % FLAGS.test_data_url)

    if FLAGS.mode == 'save_pb':
        if FLAGS.deploy_script_path == '' or FLAGS.freeze_weights_file_path == '':
            raise Exception('you must specify FLAGS.deploy_script_path '
                            'and FLAGS.freeze_weights_file_path when you want to save pb')
        if not mox.file.exists(FLAGS.deploy_script_path):
            raise Exception('FLAGS.deploy_script_path: %s is not exist' % FLAGS.deploy_script_path)
        if not mox.file.is_directory(FLAGS.deploy_script_path):
            raise Exception('FLAGS.deploy_script_path must be a directory, not a file path, %s' % FLAGS.deploy_script_path)
        if not mox.file.exists(FLAGS.freeze_weights_file_path):
            raise Exception('FLAGS.freeze_weights_file_path: %s is not exist' % FLAGS.freeze_weights_file_path)
        if mox.file.is_directory(FLAGS.freeze_weights_file_path):
            raise Exception('FLAGS.freeze_weights_file_path must be a file path, not a directory, %s ' % FLAGS.freeze_weights_file_path)
        if mox.file.exists(FLAGS.freeze_weights_file_path.rsplit('/', 1)[0] + '/model'):
            raise Exception('a model directory is already exist in ' + FLAGS.freeze_weights_file_path.rsplit('/', 1)[0]
                            + ', please rename or remove the model directory ')

    if FLAGS.mode == 'eval':
        if FLAGS.eval_weights_path == '' and FLAGS.eval_pb_path == '':
            raise Exception('you must specify FLAGS.eval_weights_path '
                            'or FLAGS.eval_pb_path when you want to evaluate a model')
        if FLAGS.eval_weights_path != '' and FLAGS.eval_pb_path != '':
            raise Exception('you must specify only one of FLAGS.eval_weights_path '
                            'and FLAGS.eval_pb_path when you want to evaluate a model')
        if FLAGS.eval_weights_path != '' and (not mox.file.exists(FLAGS.eval_weights_path)):
            raise Exception('FLAGS.eval_weights_path: %s is not exist' % FLAGS.eval_weights_path)
        if FLAGS.eval_pb_path != '' and (not mox.file.exists(FLAGS.eval_pb_path)):
            raise Exception('FLAGS.eval_pb_path: %s is not exist' % FLAGS.eval_pb_path)
        if FLAGS.eval_pb_path != '' and \
                (not mox.file.is_directory(FLAGS.eval_pb_path)) and \
                (not (FLAGS.eval_pb_path.endswith('model') or FLAGS.eval_pb_path.endswith('model\\'))):
            raise Exception('FLAGS.eval_pb_path must be a directory named model '
                            'which contain saved_model.pb and variables, %s' % FLAGS.eval_pb_path)
        if FLAGS.test_data_url == '':
            raise Exception('you must specify FLAGS.test_data_url when you want to evaluate a model')
        if not mox.file.exists(FLAGS.test_data_url):
            raise Exception('FLAGS.test_data_url: %s is not exist' % FLAGS.test_data_url)


def main(argv=None):
    check_args(FLAGS)

    # Create some local cache directories used for transfer data between local path and OBS path
    if not FLAGS.data_url.startswith('s3://'):
        FLAGS.data_local = FLAGS.data_url
    else:
        FLAGS.data_local = os.path.join(FLAGS.local_data_root, 'train_data/')
        if not os.path.exists(FLAGS.data_local):
            mox.file.copy_parallel(FLAGS.data_url, FLAGS.data_local)

            # 如果自己的模型需要加载预训练参数文件，可以先手动将参数文件从外网下载到自己的机器本地，再上传到OBS
            # 然后用下面的代码，将OBS上的预训练参数文件拷贝到 ModelArts 平台训练代码所在的目录
            # 拷贝代码格式为 mox.file.copy(src_path, dst_path)，其中dst_path不能是目录，必须是一个具体的文件名
            # mox.file.copy('s3://your_obs_path/imagenet_class_index.json',
            #               os.path.dirname(os.path.abspath(__file__)) + '/models/imagenet_class_index.json')
            # mox.file.copy('s3://your_obs_path/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            #               os.path.dirname(os.path.abspath(__file__)) + '/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        else:
            print('FLAGS.data_local: %s is already exist, skip copy' % FLAGS.data_local)
        
    if not FLAGS.train_url.startswith('s3://'):
        FLAGS.train_local = FLAGS.train_url
    else:
        FLAGS.train_local = os.path.join(FLAGS.local_data_root, 'model_snapshots/')
        if not os.path.exists(FLAGS.train_local):
            os.mkdir(FLAGS.train_local)
    
    if not FLAGS.test_data_url.startswith('s3://'):
        FLAGS.test_data_local = FLAGS.test_data_url
    else:
        FLAGS.test_data_local = os.path.join(FLAGS.local_data_root, 'test_data/')
        if not os.path.exists(FLAGS.test_data_local):
            mox.file.copy_parallel(FLAGS.test_data_url, FLAGS.test_data_local)
        else:
            print('FLAGS.test_data_local: %s is already exist, skip copy' % FLAGS.test_data_local)
    
    FLAGS.tmp = os.path.join(FLAGS.local_data_root, 'tmp/')
    if not os.path.exists(FLAGS.tmp):
        os.mkdir(FLAGS.tmp)

    if FLAGS.mode == 'train':
        from train import train_model
        train_model(FLAGS)
    elif FLAGS.mode == 'save_pb':
        from save_model import load_weights_save_pb
        load_weights_save_pb(FLAGS)
    elif FLAGS.mode == 'eval':
        from eval import eval_model
        eval_model(FLAGS)


if __name__ == '__main__':
    tf.app.run()
