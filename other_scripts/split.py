 
import os
import random
from shutil import copyfile
 
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
 
    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]
 
    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)
 
    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

for i in range(54):
    SOURCE_DIR = "./all/" + str(i) + "/"
    TRAINING_DIR = "D:/Huawei_XiAn_ResNet152/train_val/train/" + str(i) + "/"
    Validation_DIR = "D:/Huawei_XiAn_ResNet152/train_val/val/" + str(i) + "/"

    try:
        os.makedirs(TRAINING_DIR)
        os.makedirs(Validation_DIR)
    except OSError:
        pass
    
    split_size = 0.85
    split_data(SOURCE_DIR, TRAINING_DIR, Validation_DIR, split_size)