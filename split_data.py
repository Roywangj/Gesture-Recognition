import glob

import os
current_path = os.path.abspath(os.path.dirname(__file__))
root = current_path

def preprocess(mode):
    if mode == "train":
        dirs_collect = glob.glob(root+"/" + "train_gesture_data/*")
        file = open("train.txt", "w")
    elif mode == "valid":
        dirs_collect = glob.glob(root +"/"+ "test_gesture_data/*")
        file = open("test.txt", "w")
    for ges in dirs_collect:
        # print(ges)
        label = ges.split("/")[-1]
        print(label)
        images_list = glob.glob(ges + "/*")
        # print(images_list)
        for img_index in images_list:
            file.write(img_index + "," + label + "\n")
    file.close()


preprocess("train")
preprocess("valid")
