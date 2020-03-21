import cv2
import os
import numpy as np
import glob


BASE_PATH = 'GTSRB/Final_Training/Images'

def resize():
    counter = 0
    for dirname in os.listdir(BASE_PATH):
        for filename in os.listdir(os.path.join(BASE_PATH, dirname)):
            if filename.endswith(".ppm"):
                image_path = os.path.join(BASE_PATH, dirname, filename)
                im = cv2.imread(image_path)
                resized_im = cv2.resize(im, (32, 32))
                cv2.imwrite(image_path, resized_im)


def sanity_check():
    counter = 0
    for dirname in os.listdir(BASE_PATH):
        for filename in os.listdir(os.path.join(BASE_PATH, dirname)):
            if filename.endswith(".ppm"):
                image_path = os.path.join(BASE_PATH, dirname, filename)
                im = cv2.imread(image_path)
                if im.shape != (32, 32, 3):
                    counter += 1
    print(counter)


def load_dataset():
    filelist = glob.glob(f'{BASE_PATH}/*/*.ppm')
    X_train = np.array([np.array(cv2.imread(fname)) for fname in filelist])
    print(X_train.shape)



load_dataset()