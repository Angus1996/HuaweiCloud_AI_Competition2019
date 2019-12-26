import numpy as np
import cv2
import os
import shutil

'''
定义裁剪函数，四个参数分别是：
左上角横坐标x0
左上角纵坐标y0
裁剪宽度w
裁剪高度h
'''
crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

'''
随机裁剪
area_ratio为裁剪画面占原画面的比例
hw_vari是扰动占原高宽比的比例范围
'''
def random_crop(img, area_ratio, hw_vari):
    h, w = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta
	
	# 下标进行裁剪，宽高必须是正整数
    w_crop = int(round(w*np.sqrt(area_ratio*hw_mult)))
	
	# 裁剪宽度不可超过原图可裁剪宽度
    if w_crop > w:
        w_crop = w
		
    h_crop = int(round(h*np.sqrt(area_ratio/hw_mult)))
    if h_crop > h:
        h_crop = h
	
	# 随机生成左上角的位置
    x0 = np.random.randint(0, w-w_crop+1)
    y0 = np.random.randint(0, h-h_crop+1)
	
    return crop_image(img, x0, y0, w_crop, h_crop)

'''
定义旋转函数：
angle是逆时针旋转的角度
crop是个布尔值，表明是否要裁剪去除黑边
'''
def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
	
	# 旋转角度的周期是360°
    angle %= 360
	
	# 用OpenCV内置函数计算仿射矩阵
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
	
	# 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))

	# 如果需要裁剪去除黑边
    if crop:
	    # 对于裁剪角度的等效周期是180°
        angle_crop = angle % 180
		
		# 并且关于90°对称
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
			
		# 转化角度为弧度
        theta = angle_crop * np.pi / 180.0
		
		# 计算高宽比
        hw_ratio = float(h) / float(w)
		
		# 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
		
		# 计算分母项中和宽高比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
		
		# 计算分母项
        denominator = r * tan_theta + 1
		
		# 计算最终的边长系数
        crop_mult = numerator / denominator
		
		# 得到裁剪区域
        w_crop = int(round(crop_mult*w))
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated

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
            img_rotated = rotate_image(img, +90, False)
            cv2.imwrite('./rotated_data_+90/rotated3_'+img_item, img_rotated)
        else:
            txt_path = os.path.join(base_dir, img_item)
            img_name, label = open_txt(txt_path)
            new_txt = open('./rotated_data_+90/rotated3_'+img_item, 'w')
            new_txt.write('rotated3_'+img_name+','+label+'\n')
            # shutil.copyfile(txt_path,'./rotated_data_-10/rotated_'+img_item)
            new_txt.close()
