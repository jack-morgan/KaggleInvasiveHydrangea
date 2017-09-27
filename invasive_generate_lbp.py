import pandas as pd
import cv2
import os
import numpy as np
from skimage import feature

# this just loads the train_labels file that was provided by kaggle
labels_csv = pd.read_csv('train_labels.csv')
# this returns a list of paths of all the files in the folder 'train'
# (also provided by kaggle), this folder contains the training images.
img_paths = os.listdir('train/')


# this function was created to put separate an array with the images and its corresponding labels...
def load_train_images(img_paths, labels_csv):
    # empty list to hold images and their labels... note that they should match in their indices...
    # i.e. image[0] must be corresponded to label[0], this is important.
    images = []
    labels = []
    # loop through the list of paths to each image...
    # we know that the paths are set like so.. 1.jpg, 2.jpg, 3.jpg..sequentially...soo
    for i in img_paths:
        # i is a string with the path of each image: '1.jpg'
        # i.split('.') => ['1', 'jpg']
        # I want the first element of the list, that's why there's a [0]
        # row = 1, on the first iteration...
        # however, the labels_csv dataframe starts from index 0...that's why I subtracted 1
        # to index its rows...
        # so the class of 1.jpg is labels_csv[1-1][column]...since we can index columns
        # as labels on dataframes, the column name is 'invasive'...so that's the column we want.
        # please look the csv file to see how it's written.
        row = int(i.split('.')[0])
        forest_class = labels_csv.iloc[row - 1]['invasive']

        # read each image with opencv and convert it to grayscale ...
        # its not necessary to convert it like this...you can also read the image
        # in its original colospace and convert it with cv2.cvtColor() method.
        img = cv2.imread('train/' + i, cv2.IMREAD_GRAYSCALE)
        # IMPORTANT!
        # I resized every image into and arbitrary size to make processing time lower.
        # this affects classification rate. (you may take note that this is one of the parameters,
        # that you'll have to test and see how affects your results)
        img_resized = cv2.resize(img, dsize=(500, 300))

        # append loaded images to iamges list and corresponding labels to labels list
        # now I have images[0] = 1.jpg and labels[0] = 1.jpg class, desired output, Y.
        images.append(img_resized)
        labels.append(forest_class)

    return images, labels


# this function will take the list of images we created previously
# and compute the LBP histogram of each image and return a new list with the
# corresponding LBP features of each image.
def generate_lbp(images, num_points, radius):
    # data to be returned
    data = []

    # loop through the images list using enumerate because it returns the element
    # of the list (img) as well as the index of the corresponding image (i)
    # I only need the index i to print the current image to follow the process along
    for i, img in enumerate(images):
        # this line computes the LBP (num_points, radius) of the image...
        # all you have to know is that the LBP variable is an array, just like an image, with
        # its values altered according to the LBP method.
        # IMPORTANT AGAIN:
        # num_points and radius are parameters that also affect (greatly) classification
        # performance. We'll need to test different configurations of it.
        # the values I chose to test were an "educated guess", but also arbitrary based
        # on image size.
        lbp = feature.local_binary_pattern(img, num_points, radius, method='uniform')
        # the image itself is not the features vector, the histogram of each LBP code
        # generated from the image is. This is what this line does, generated the
        # histogram of the LBP generated image.
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))

        # convert histogram to float values.
        hist = hist.astype('float')
        # normalize it... to make it between 0 and 1 [0,1]
        hist /= hist.sum()

        # I just printed the index so follow in the console the progress.
        print(i)

        # for debug purposes if you want to see what the LBP image looks like.
        # (they're interesting)

        # cv2.imshow('original image', img)
        # cv2.imshow('lbp image', lbp)
        # cv2.waitKey(1)

        # final data list to return. (feature vector)
        data.append(hist)

    return data


''' generating training data '''
print('loading images and labels...')
train_images, labels_list = load_train_images(img_paths, labels_csv)
print('generating training data...')
# generating LBP with arbitrary (kinda) num_points and radius parameters
train_data = generate_lbp(train_images, 24, 8)

# converting my list to a DataFrame (I wanted to use data frames with the sklearn framework)
train_data = pd.DataFrame(train_data)
# append the labels column to the features vector
train_data = pd.concat([train_data, pd.DataFrame(labels_list)], axis=1)
# writing the result data to csv (to train the classifiers on another file)
train_data.to_csv('train_data.csv', index=False)

# follow along..
#l = range(3)  # l = [0,1,2]
#for i in range(3):
#print(i)  # -> 0,1,2
# this is the same as....
#for i in l:
 #   print(i)

#for i, value in enumerate(l):
#    print(i, value)
    # this will give i = 0 (first index) and value = 0 (first value of the list)
    # OH OK - so enumerate will return both these values - index and value. I understand now :)
    # Let's keep this here so I can remember in the future - thanks
