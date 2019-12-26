from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as TF
import torch
import skimage
import random
import numpy as np

# 读取一张测试图片
path = "./img_4546.jpg"
img = Image.open(path)

def add_noise(img):
    a = random.random()
    if a > 0.7:
        img=np.array(img)
        result = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True)
        return Image.fromarray(np.uint8(result*255))
    elif a < 0.7 and a > 0.3:
        img=np.array(img)
        #随机生成5000个椒盐
        rows,cols,dims=img.shape
        for i in range(10000):
            x=np.random.randint(0,rows)
            y=np.random.randint(0,cols)
            img[x,y,:]=255
        for i in range(10000):
            x=np.random.randint(0,rows)
            y=np.random.randint(0,cols)
            img[x,y,:]=0
        return Image.fromarray(np.uint8(img))
    else:
        return img

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])

transform = transforms.Compose([
    # transforms.Pad(padding=250),
    transforms.Resize(240),
    transforms.CenterCrop(224),
    # transforms.RandomResizedCrop(224, scale=(0.99, 1.0)),
    # transforms.Resize((224,224)),
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    # transforms.RandomAffine(degrees=90, translate=(0, 0.5), scale=(0.8, 1.2), shear=(10, 30), fillcolor=0),
    # transforms.Lambda(add_noise)
])

new_img = transform(img)
new_img.save('./result2.jpg')

