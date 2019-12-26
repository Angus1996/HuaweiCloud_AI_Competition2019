from PIL import Image
import os
base_dir = './train_val_v4/'

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
            dst_place = './train_val_v6/' + item_cate + '/' + img_folder + '/' + img_name.split('.')[0] + '.jpg'
            if not os.path.exists('./train_val_v6/' + item_cate + '/' + img_folder + '/'):
                os.makedirs('./train_val_v6/' + item_cate + '/' + img_folder + '/')
            img.save(dst_place)