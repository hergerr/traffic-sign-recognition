import cv2
import os
import numpy as np
import glob
import csv
from matplotlib import pyplot as plt
import random

BASE_PATH = 'GTSRB/Final_Training/Images'
TEST_PATH = 'GTSRB/Final_Test'

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
    testlist = glob.glob(f'{TEST_PATH}/Images/*.ppm')
    X_train = np.array( [np.array( cv2.imread(fname) ) for fname in filelist] )
    X_test = np.array([np.array( cv2.imread(fname) ) for fname in testlist])

    Y_train = np.array([np.zeros(43) for fname in filelist])
    for i,fname in enumerate(filelist):
        Y_train[i][int(fname.split('/')[3])]=1

    with open(f'{TEST_PATH}/GT-final_test.csv') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=';')
        data = list(spamreader)
        data.pop(0)

    Y_test = np.array([np.zeros(43) for fname in testlist])
    for i,fname in enumerate(testlist):
        image_name = fname.split('/')[3]
        image_number = int(image_name.split('.')[0])
        classid = int(data[image_number][7])
        Y_test[i][classid]=1

    return X_train, Y_train, X_test, Y_test

def show_image(index, X, Y):
    plt.imshow(cv2.cvtColor(X[index],cv2.COLOR_BGR2RGB))
    plt.show()
    print('Belongs to class nr: ' + str(int(np.where(Y[index]==1)[0])))


#resize()
#sanity_check()
X_train, Y_train, X_test, Y_test = load_dataset()
show_image(random.randint(0,12629),X_train, Y_train])