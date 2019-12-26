import numpy as np
import random
import cv2
import os

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.0001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out


def open_txt(txt_path):
    txt_file = open(txt_path, 'r', encoding='utf-8')
    lines = txt_file.readlines()[0]
    lines = lines.strip('\n')
    img_name, label = lines.split(',')[0], lines.split(',')[1]
    txt_file.close()
    return img_name, label

if __name__ == "__main__":
    base_dir = './train_data/'
    for img_item in os.listdir(base_dir):
        print("[INFO] Processing"+img_item)
        if img_item.split('.')[-1] != 'txt':
            img_path = os.path.join(base_dir, img_item)
            img = cv2.imread(img_path)
            noised = gasuss_noise(img)
            cv2.imwrite('./noised_train_data_v2/noised_'+img_item, noised)
        else:
            txt_path = os.path.join(base_dir, img_item)
            img_name, label = open_txt(txt_path)
            new_txt = open('./noised_train_data_v2/noised_'+img_item, 'w')
            new_txt.write('noised_'+img_name+','+label+'\n')
            # shutil.copyfile(txt_path,'./rotated_data_-10/rotated_'+img_item)
            new_txt.close()

    # test_img = cv2.imread('rotated_img_1.jpg')
    # noised = gasuss_noise(test_img)
    # cv2.imwrite('noised_img_1.jpg', noised)

    # noise = sp_noise(test_img, 0.005)
    # out = test_img + noise
    # if out.min() < 0:
    #     low_clip = -1.
    # else:
    #     low_clip = 0.
    # out = np.clip(out, low_clip, 1.0)
    # out = np.uint8(out*255)
    # cv2.imwrite('noised_img_1.jpg', out)