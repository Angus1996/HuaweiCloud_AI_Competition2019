# -*- coding: utf-8 -*-
"""
ModelArts notebook 的作用是用于代码调试，因此也可以调试推理脚本
先在 notebook 上跑通推理脚本，再将其修改成“部署上线”功能需要的customize_service.py
可以对比本文件 inference.py 和 customize_service.py，查看有哪些修改之处
"""
import os
import codecs
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from tensorflow.python.saved_model import tag_constants
# from model_service.tfserving_model_service import TfServingBaseService

# import time
# from metric.metrics_manager import MetricsManager
# import log
# logger = log.getLogger(__name__)


class ImageClassificationService():
    def __init__(self, model_name, model_path):
        if self.is_tf_gpu_version() is True:
            print('use tf GPU version,', tf.__version__)
        else:
            print('use tf CPU version,', tf.__version__)

        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        self.signature_key = 'predict_images'

        self.input_size = 224  # the input image size of the model

        # add the input and output key of your pb model here,
        # these keys are defined when you save a pb file
        self.input_key_1 = 'input_img'
        self.output_key_1 = 'output_score'
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.get_default_graph().as_default():
            self.sess = tf.Session(graph=tf.Graph(), config=config)
            meta_graph_def = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], self.model_path)
            self.signature = meta_graph_def.signature_def

            # define input and out tensor of your model here
            input_images_tensor_name = self.signature[self.signature_key].inputs[self.input_key_1].name
            output_score_tensor_name = self.signature[self.signature_key].outputs[self.output_key_1].name
            self.input_images = self.sess.graph.get_tensor_by_name(input_images_tensor_name)
            self.output_score = self.sess.graph.get_tensor_by_name(output_score_tensor_name)

        self.label_id_name_dict = \
            {
                "0": "工艺品/仿唐三彩",
                "1": "工艺品/仿宋木叶盏",
                "2": "工艺品/布贴绣",
                "3": "工艺品/景泰蓝",
                "4": "工艺品/木马勺脸谱",
                "5": "工艺品/柳编",
                "6": "工艺品/葡萄花鸟纹银香囊",
                "7": "工艺品/西安剪纸",
                "8": "工艺品/陕历博唐妞系列",
                "9": "景点/关中书院",
                "10": "景点/兵马俑",
                "11": "景点/南五台",
                "12": "景点/大兴善寺",
                "13": "景点/大观楼",
                "14": "景点/大雁塔",
                "15": "景点/小雁塔",
                "16": "景点/未央宫城墙遗址",
                "17": "景点/水陆庵壁塑",
                "18": "景点/汉长安城遗址",
                "19": "景点/西安城墙",
                "20": "景点/钟楼",
                "21": "景点/长安华严寺",
                "22": "景点/阿房宫遗址",
                "23": "民俗/唢呐",
                "24": "民俗/皮影",
                "25": "特产/临潼火晶柿子",
                "26": "特产/山茱萸",
                "27": "特产/玉器",
                "28": "特产/阎良甜瓜",
                "29": "特产/陕北红小豆",
                "30": "特产/高陵冬枣",
                "31": "美食/八宝玫瑰镜糕",
                "32": "美食/凉皮",
                "33": "美食/凉鱼",
                "34": "美食/德懋恭水晶饼",
                "35": "美食/搅团",
                "36": "美食/枸杞炖银耳",
                "37": "美食/柿子饼",
                "38": "美食/浆水面",
                "39": "美食/灌汤包",
                "40": "美食/烧肘子",
                "41": "美食/石子饼",
                "42": "美食/神仙粉",
                "43": "美食/粉汤羊血",
                "44": "美食/羊肉泡馍",
                "45": "美食/肉夹馍",
                "46": "美食/荞面饸饹",
                "47": "美食/菠菜面",
                "48": "美食/蜂蜜凉粽子",
                "49": "美食/蜜饯张口酥饺",
                "50": "美食/西安油茶",
                "51": "美食/贵妃鸡翅",
                "52": "美食/醪糟",
                "53": "美食/金线油塔"
            }

    def is_tf_gpu_version(self):
        from tensorflow.python.client import device_lib
        is_gpu_version = False
        devices_info = device_lib.list_local_devices()
        for device in devices_info:
            if 'GPU' == str(device.device_type):
                is_gpu_version = True
                break

        return is_gpu_version

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = img.resize((self.input_size, self.input_size))
                img = np.asarray(img, dtype=np.float32)
                img = preprocess_input(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data[self.input_key_1]
        img = img[np.newaxis, :, :, :]  # the input tensor shape of resnet is [?, 224, 224, 3]
        pred_score = self.sess.run([self.output_score], feed_dict={self.input_images: img})
        if pred_score is not None:
            pred_label = np.argmax(pred_score[0], axis=1)[0]
            result = {'result': self.label_id_name_dict[str(pred_label)]}
        else:
            result = {'result': 'predict score is None'}
        return result

    def _postprocess(self, data):
        return data

    # def inference(self, data):
    #     '''
    #     Wrapper function to run preprocess, inference and postprocess functions.
    #
    #     Parameters
    #     ----------
    #     data : map of object
    #         Raw input from request.
    #
    #     Returns
    #     -------
    #     list of outputs to be sent back to client.
    #         data to be sent back
    #     '''
    #     pre_start_time = time.time()
    #     data = self._preprocess(data)
    #     infer_start_time = time.time()
    #     # Update preprocess latency metric
    #     pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
    #     logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
    #
    #     if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
    #         MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
    #
    #     data = self._inference(data)
    #     infer_end_time = time.time()
    #     infer_in_ms = (infer_end_time - infer_start_time) * 1000
    #
    #     logger.info('infer time: ' + str(infer_in_ms) + 'ms')
    #     data = self._postprocess(data)
    #
    #     # Update inference latency metric
    #     post_time_in_ms = (time.time() - infer_end_time) * 1000
    #     logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
    #     if self.model_name + '_LatencyInference' in MetricsManager.metrics:
    #         MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
    #
    #     # Update overall latency metric
    #     if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
    #         MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
    #
    #     logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
    #     data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
    #     return data


def infer_on_dataset(img_dir, label_dir, model_path):
    if not os.path.exists(img_dir):
        print('img_dir: %s is not exist' % img_dir)
        return None
    if not os.path.exists(label_dir):
        print('label_dir: %s is not exist' % label_dir)
        return None
    if not os.path.exists(model_path):
        print('model_path: %s is not exist' % model_path)
        return None
    output_dir = model_path + '_output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    infer = ImageClassificationService('', model_path)
    files = os.listdir(img_dir)
    error_results = []
    right_count = 0
    total_count = 0
    for file_name in files:
        if not file_name.endswith('jpg'):
            continue

        with codecs.open(os.path.join(label_dir, file_name.split('.jpg')[0] + '.txt'), 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_name.split('.jpg')[0] + '.txt'))
            continue
        gt_label = infer.label_id_name_dict[line_split[1]]

        img_path = os.path.join(img_dir, file_name)
        img = Image.open(img_path)
        img = img.resize((infer.input_size, infer.input_size))
        img = np.asarray(img, dtype=np.float32())
        img = preprocess_input(img)
        result = infer._inference({"input_img": img})
        pred_label = result.get('result', 'error')

        total_count += 1
        if pred_label == gt_label:
            right_count += 1
        else:
            error_results.append(', '.join([file_name, gt_label, pred_label]) + '\n')

    acc = float(right_count) / total_count
    result_file_path = os.path.join(output_dir, 'accuracy.txt')
    with codecs.open(result_file_path, 'w', 'utf-8') as f:
        f.write('# predict error files\n')
        f.write('####################################\n')
        f.write('file_name, gt_label, pred_label\n')
        f.writelines(error_results)
        f.write('####################################\n')
        f.write('accuracy: %s\n' % acc)
    print('accuracy result file saved as %s' % result_file_path)
    print('accuracy: %0.4f' % acc)
    return acc, result_file_path


if __name__ == '__main__':
    img_dir = r'/home/ma-user/work/xi_an_ai/datasets/test_data'
    label_dir = r'/home/ma-user/work/xi_an_ai/datasets/test_data'
    model_path = r'/home/ma-user/work/xi_an_ai/model_snapshots/V0003/model'
    infer_on_dataset(img_dir, label_dir, model_path)
