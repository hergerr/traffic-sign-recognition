import cv2
import os
import numpy as np
import glob
import csv
from matplotlib import pyplot as plt
import random

BASE_PATH = 'GTSRB/Final_Training/Images'
TEST_PATH = 'GTSRB/Final_Test'

class_names = {
    0: "Speed limit to 20",
    1: "Speed limit to 30",
    2: "Speed limit to 50",
    3: "Speed limit to 60",
    4: "Speed limit to 70",
    5: "Speed limit to 80",
    6: "End of speed limit up to 80",
    7: "Speed limit to 100",
    8: "Speed limit to 120",
    9: "No overtaking",
    10: "No overtaking for trucks",
    11: "Priority to through-traffic at the next intersection/crossroads only",
    12: "Priority Road",
    13: "Yield to cross traffic",
    14: "Stop and give way",
    15: "No vehicles of any kind permitted",
    16: "No trucks permitted",
    17: "Do not enter",
    18: "Danger point",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curves, first to left",
    22: "Uneven surfaces ahead",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Roadworks",
    26: "Traffic signals - Primary Priority",
    27: "Pedestrian crossing",
    28: "Children crossing",
    29: "Bicycle lane",
    30: "Snow or Ice possible ahead",
    31: "Wild animals possible",
    32: "End of previous limitation",
    33: "You must turn right ahead",
    34: "You must turn left ahead",
    35: "You must go straight ahead",
    36: "You must go straight or turn right",
    37: "You must go straight or turn left",
    38: "Keep right of traffic barrier/divider",
    39: "Keep left of traffic barrier/divider",
    40: "Roundabout",
    41: "End of the ban on overtaking",
    42: "End of the ban on overtaking for trucks",
    43: "Unclassified"
}

def resize():
    counter = 0
    for dirname in os.listdir(BASE_PATH):
        for filename in os.listdir(os.path.join(BASE_PATH, dirname)):
            if filename.endswith(".ppm"):
                image_path = os.path.join(BASE_PATH, dirname, filename)
                im = cv2.imread(image_path)
                resized_im = cv2.resize(im, (32, 32))
                cv2.imwrite(image_path, resized_im)

    testlist = glob.glob(f'{TEST_PATH}/Images/*.ppm')
    for filename in testlist:
        im = cv2.imread(filename)
        resized_im = cv2.resize(im, (32,32))
        cv2.imwrite(filename, resized_im)


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
    print('Sign meanings:\n ' + class_names[ int(np.where(Y[index]==1)[0])] )


resize()
sanity_check()
# X_train, Y_train, X_test, Y_test = load_dataset()
# permutation = list(np.random.permutation(43))
# shuffled_X = X_train[2, :, :, :]
# #print(shuffled_X)
# show_image(random.randint(0,12629),X_test, Y_test)

def resize(basic_path):
    counter = 0
    for filename in os.listdir(basic_path):
        if filename.endswith(".png"):
            image_path = os.path.join(basic_path, filename)
            im = cv2.imread(image_path)
            resized_im = cv2.resize(im, (32, 32))
            cv2.imwrite(image_path, resized_im)

#resize('GTSRB/Polish_Test')

def break_image(image):
    stainColor = [139, 69, 19]
    stainRay = 4
    rowLen = 3
    X=16 + random.randint(-5,5)
    Y=8 + random.randint(-5,5)
    
    Y -= stainRay
    for i in range(stainRay + 1):
        Xtemp = X
        for j in range(rowLen):
            image[Y, Xtemp] = stainColor
            Xtemp+=1
        
        X -= 1
        Y+=1
        rowLen += 2

    for i in range(stainRay+2):
        Xtemp = X
        for j in range(rowLen):
            image[Y, Xtemp] = stainColor
            Xtemp+=1
        X += 1
        Y+=1
        rowLen -=2

    #return image


image_to_break = cv2.imread('znak6.png')
image_to_break = cv2.cvtColor(image_to_break, cv2.COLOR_BGR2RGB)
break_image(image_to_break)
plt.imshow(image_to_break)
plt.show()