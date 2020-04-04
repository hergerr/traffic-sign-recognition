import cv2
import os
import numpy as np
import glob
import csv
from matplotlib import pyplot as plt
import random

BASE_PATH = 'GTSRB/Final_Training/Images'
TEST_PATH = 'GTSRB/Final_Test'

class_sizes = {}
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


def count_data_train_set_size():

    for i,dirname in enumerate(os.listdir(BASE_PATH)):
        curr_path = os.listdir(os.path.join(BASE_PATH, dirname))
        files = [name for name in curr_path if os.path.isfile(os.path.join(BASE_PATH, dirname, name))]
        length = len(files) - 1#you have to subtract csv file in every class
        class_sizes[dirname] = length
        print('Class ' + dirname + ': ' + str(length))
    # print(class_sizes)

    
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def augument_1_img_to_10(image_name):
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #rottating by -5, and 5 degree, _5 == -5
    img_5 = rotate_image(img, -5)
    img5 = rotate_image(img, 5)
    img_10 = rotate_image(img, -10)
    img10 = rotate_image(img, 10)

    #changing brightnes of images
    img_5_dark = cv2.cvtColor(img_5, cv2.COLOR_RGB2HSV)
    img_5_dark[:,:,2] = np.clip(img_5_dark[:,:,2]*0.4, a_min=0,a_max=255)
    img_5_dark = cv2.cvtColor(img_5_dark, cv2.COLOR_HSV2RGB) 

    img_5_light = cv2.cvtColor(img_5, cv2.COLOR_RGB2HSV)
    img_5_light[:,:,2] = np.clip(img_5_light[:,:,2]*1.2, a_min=0,a_max=255)
    img_5_light = cv2.cvtColor(img_5_light, cv2.COLOR_HSV2RGB) 

    img5_dark = cv2.cvtColor(img_5, cv2.COLOR_RGB2HSV)
    img5_dark[:,:,2] = np.clip(img5_dark[:,:,2]*0.4, a_min=0,a_max=255) 
    img5_dark = cv2.cvtColor(img5_dark, cv2.COLOR_HSV2RGB) 

    img5_light = cv2.cvtColor(img_5, cv2.COLOR_RGB2HSV)
    img5_light[:,:,2] = np.clip(img5_light[:,:,2]*1.2, a_min=0,a_max=255) 
    img5_light = cv2.cvtColor(img5_light, cv2.COLOR_HSV2RGB) 

    img_10_dark = cv2.cvtColor(img_10, cv2.COLOR_RGB2HSV)
    img_10_dark[:,:,2] = np.clip(img_10_dark[:,:,2]*0.4, a_min=0,a_max=255) 
    img_10_dark = cv2.cvtColor(img_10_dark, cv2.COLOR_HSV2RGB) 

    img_10_light = cv2.cvtColor(img_10, cv2.COLOR_RGB2HSV)
    img_10_light[:,:,2] = np.clip(img_10_light[:,:,2]*1.2, a_min=0,a_max=255) 
    img_10_light = cv2.cvtColor(img_10_light, cv2.COLOR_HSV2RGB) 

    # fig, ax = plt.subplots(4,3)
    # ax[0, 0].imshow(img)
    
    # ax[1, 0].imshow(img_5)
    # ax[1, 1].imshow(img5)
    # ax[1, 2].imshow(img_10)
    # ax[2, 0].imshow(img10)
    # ax[2, 1].imshow(img_5_dark)
    # ax[2, 2].imshow(img_5_light)
    # ax[3, 0].imshow(img5_dark)
    # ax[3, 1].imshow(img5_light)
    # ax[3, 2].imshow(img_10_dark)

    # plt.show()
    cv2.imwrite(image_name.split('.')[0]+'img_5.ppm', cv2.cvtColor(img_5,cv2.COLOR_RGB2BGR)) 
    cv2.imwrite(image_name.split('.')[0]+'img5.ppm', cv2.cvtColor(img5,cv2.COLOR_RGB2BGR)) 
    cv2.imwrite(image_name.split('.')[0]+'img_10.ppm', cv2.cvtColor(img_10,cv2.COLOR_RGB2BGR)) 
    cv2.imwrite(image_name.split('.')[0]+'img10.ppm', cv2.cvtColor(img10,cv2.COLOR_RGB2BGR)) 
    cv2.imwrite(image_name.split('.')[0]+'img_5_dark.ppm', cv2.cvtColor(img_5_dark,cv2.COLOR_RGB2BGR)) 
    cv2.imwrite(image_name.split('.')[0]+'img_5_light.ppm', cv2.cvtColor(img_5_light,cv2.COLOR_RGB2BGR)) 
    cv2.imwrite(image_name.split('.')[0]+'img5_dark.ppm', cv2.cvtColor(img5_dark,cv2.COLOR_RGB2BGR)) 
    cv2.imwrite(image_name.split('.')[0]+'img5_light.ppm', cv2.cvtColor(img5_light,cv2.COLOR_RGB2BGR)) 
    cv2.imwrite(image_name.split('.')[0]+'img_10_dark.ppm', cv2.cvtColor(img_10_dark,cv2.COLOR_RGB2BGR)) 
    cv2.imwrite(image_name.split('.')[0]+'img_10_light.ppm', cv2.cvtColor(img_10_light,cv2.COLOR_RGB2BGR)) 


def augument_train_set():#every train class will have 2300 elements
    for i,dirname in enumerate(os.listdir(BASE_PATH)):
        curr_path = os.listdir(os.path.join(BASE_PATH, dirname))
        files = [name for name in curr_path if name.find('.csv') == -1]
        #print(files)
        to_augument =[]
        for n in range(int((1800-len(files))/10)):#tyle ma być nowych img w danej klasie
            while True:
                elem = random.choice(files)
                if( elem not in to_augument):
                    to_augument.append(elem)
                    break
        #print(to_augument)
        
        [augument_1_img_to_10(os.path.join(BASE_PATH,dirname , image)) for image in to_augument]


def scale_down_train_set_to_210_imgs():
    for dirname in os.listdir(BASE_PATH):
        curr_path = os.listdir(os.path.join(BASE_PATH, dirname))
        files = [os.path.join(BASE_PATH, dirname, name) for name in curr_path]
        files = files[211:]
        for img_to_delete in files:
            os.remove(img_to_delete)



# augument_1_img_to_10('test.ppm')  

#resize()
#sanity_check()
# X_train, Y_train, X_test, Y_test = load_dataset()
# permutation = list(np.random.permutation(43))
# shuffled_X = X_train[2, :, :, :]
# #print(shuffled_X)
# show_image(random.randint(0,12629),X_test, Y_test)
count_data_train_set_size()
#augument_train_set()
scale_down_train_set_to_210_imgs()
count_data_train_set_size()

#X_train, Y_train, X_test, Y_test = load_dataset()
#print(X_train.shape)
#(m, n_H0, n_W0, n_C0) = X_train.shape