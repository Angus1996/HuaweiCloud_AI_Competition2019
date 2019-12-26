from PIL import Image
import os

class Resize(object):
    def __init__(self, size, interpolation = Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h*ratio)
            w_padding = (t-w)//2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w*ratio)
            h_padding = (t-h)//2
            # print(h_padding)
            # print(w, h)
            img = img.crop((0, -h_padding, w, h+h_padding))
        
        img = img.resize(self.size, self.interpolation)

        return img

base_dir = './train_val_v7/'
# img = Image.open('./crawled_1.jpg')
resize = Resize([256,256])

for item_cate in os.listdir(base_dir):
    second_dir = os.path.join(base_dir, item_cate)
    # for i in range(3,19):
    for img_folder in os.listdir(second_dir):
        # img_folder = str(i)
        third_folder = os.path.join(second_dir, img_folder)
        for img_name in os.listdir(third_folder):
            img_full_path = os.path.join(third_folder, img_name)
            print(img_full_path)
            img = Image.open(img_full_path)
            dst_place = './padded_train_val/' + item_cate + '/' + img_folder + '/' + img_name.split('.')[0] + '.jpg'
            if not os.path.exists('./padded_train_val/' + item_cate + '/' + img_folder + '/'):
                os.makedirs('./padded_train_val/' + item_cate + '/' + img_folder + '/')
            resized_img = resize.__call__(img)
            resized_img.save(dst_place)
#             img.save(dst_place)

# resized_img = resize.__call__(img)
# resized_img.save('./test.jpg')