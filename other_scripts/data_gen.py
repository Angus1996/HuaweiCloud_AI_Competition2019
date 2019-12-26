from keras.preprocessing.image import ImageDataGenerator
import os
import shutil

train_base_dir = './train_val/train'
val_base_dir = './train_val/val'
img_width = 256
img_height = 256
batch_size = 20

datagen = ImageDataGenerator(
        # shear_range=0.1,
        zoom_range=0.1,
        brightness_range=(0.9,1.1),
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

for i in range(54):
    train_data_dir = os.path.join(train_base_dir, str(i))
    save_dir = 'D:/Contest/HuaWei_Cloud_AI/augment_train_val_v4/train/' + str(i)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle = True,
            save_to_dir = save_dir,
            save_prefix = 'aug',
            # classes = FishNames,
            class_mode = 'categorical')
    for img in os.listdir(os.path.join(train_data_dir, str(i))):
        shutil.copy(os.path.join(train_data_dir, str(i), img), os.path.join(save_dir, img))
    while(True):
        if len(os.listdir(save_dir)) > 600 :
            break
        generator.next()
        
        