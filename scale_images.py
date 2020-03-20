import cv2
import os


def resize():
    base_path = 'GTSRB/Final_Training/Images'
    counter = 0
    for dirname in os.listdir(base_path):
        for filename in os.listdir(os.path.join(base_path, dirname)):
            if filename.endswith(".ppm"):
                image_path = os.path.join(base_path, dirname, filename)
                im = cv2.imread(image_path)
                resized_im = cv2.resize(im, (32, 32))
                cv2.imwrite(image_path, resized_im)


def sanity_check():
    base_path = 'GTSRB/Final_Training/Images'
    counter = 0
    for dirname in os.listdir(base_path):
        for filename in os.listdir(os.path.join(base_path, dirname)):
            if filename.endswith(".ppm"):
                image_path = os.path.join(base_path, dirname, filename)
                im = cv2.imread(image_path)
                if im.shape != (32, 32, 3):
                    counter += 1
    print(counter)


# resize()
# sanity_check()
