#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageNet - Cat's Classifiers
Created on Sat May 11 20:25:36 2019

@author: GounariOlympia
https://www.learnopencv.com/keras-tutorial-using-pre-trained-imagenet-models/
"""

from tensorflow.python.keras.preprocessing import image
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import cv2
import urllib
import os


def url_to_image(url):
     '''
     Download the image, convert it to a NumPy array, and then read it into OpenCV format'''
     resp = urllib.request.urlopen(url)
     image = np.asarray(bytearray(resp.read()), dtype="uint8")
     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
     # return the image
     return(image)

    
def save_images_from_wnid(cwd, categ_name, wnid, n_of_images):
    '''
    For given name and wnid, of ImageNet, create folder in cwd and save as many images as
    asked (n_of_images). If something goes wrong with one image's url, the next image
    will be dowloaded (and so on), just to be sure that number of downloaded images = n_of_images'''
    
    page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+str(wnid)) # synset
    # Puts the content of the website into the soup variable, each url on a different line
    soup = BeautifulSoup(page.content, 'html.parser')
    str_soup = str(soup)    # Convert soup to string so it can be split
    split_urls = str_soup.split('\r\n')   # Split so each url is a different possition on a list
    print("\nTotal urls for category {} are {}".format(categ_name, len(split_urls)))
    #img_rows, img_cols = 32, 32    # Number of rows and columns to convert the images to             NOT USED !
    #input_shape = (img_rows, img_cols, 3)   # Format to store the images (rows, columns, bands)
    
    try:  
        os.mkdir(str(categ_name))    # Create forlder to save images for every class
        print("Directory "+str(categ_name)+" created...\n")
        # Store asked images in the created directory
        dirpath = str(cwd)+'/'+str(categ_name)  # Dirpath for the created directory
        for i in range(len(split_urls)):     # For every url, of total urls
            # Check if url works, and check number of saved images in the created folder
            if not split_urls[i] == None and len(os.listdir(dirpath)) < n_of_images:
                try:
                    I = url_to_image(split_urls[i])
                    if (len(I.shape))==3:    # Check image's tensor rank
                        save_path = str(cwd)+'/'+str(categ_name)+'/img'+str(i)+'.jpg'  # Create a name of each image
                        cv2.imwrite(save_path, I)
                        print("Image {} with shape {}...saved!".format(i, I.shape))
                except:
                    print('Exception occured for {}th image..... Continue to the next one...'.format(i))
                    pass
    except FileExistsError:  
        print ("Directory "+str(categ_name)+" already exists !\n")
        pass
    return(0)


def predictNevaluate(model, target_size, cwd, categ_names, wnids_list):
    '''
    target_size is a tuple of image-dimensions expected from every model
    wnids_list is nested list with url's ending for each category, and the corresponding categ name.
    wnids_list and categ_names must have the same sequnce, and matters to confusion matrix computation.
    categ_names is list of categories. Their sequnce matters to confusion matrix computation.
    Returns a dataframe
    
    '''
# Create Confusion Matrix
    # Create dataframe full of zeros to store confusion matrix
    cm = pd.DataFrame(np.zeros((len(wnids_list)+1, len(wnids_list)+2)))
    # Create column-names for confusion matrix
    cm.columns = categ_names+['Other', 'Precision-Producer-Accur']  # Sorted categories
    # Create row-names for confusion matrix
    cm.index = categ_names+['Recall-User-Reliab']
        
    for i in range(len(wnids_list)):
        img_path = cwd +'/'+ wnids_list[i][0]
        print('\n\nImages from folder {}:'.format(wnids_list[i][0]))
        for root, subfolders, files in os.walk(img_path):
            for file in files:
                # load an image in PIL format
                img = image.load_img(img_path+'/'+file, target_size=target_size)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                preds = model.predict(x)
                # Top three predictions for every image
                imRes = decode_predictions(preds, top=3)[0]
                # decode the results into a list of tuples (class, description, probability)
                print('\nFile {} predicted:\n{}'.format(file, imRes))
                
            # Create confusion matrix
                # If name of predicted category is same as folder. Compare strings
                if imRes[0][1] == wnids_list[i][0]:
                    # Add 1 at value in cell with row same as folder name
                    whichRow = np.where(cm.index == wnids_list[i][0])
                    # and col as predicted-name. [col][row]
                    cm[imRes[0][1]][whichRow[0][0]] += 1
                    print('TRUE\n')
                # If is different from folder but is one of the five categories
                elif (imRes[0][1] in categ_names and imRes[0][1] != wnids_list[i][0]):
                    # Add 1 at value in cell with row same as folder name
                    whichRow = np.where(cm.index == wnids_list[i][0])
                    # and col as predicted-name.
                    cm[imRes[0][1]][whichRow[0][0]] += 1
                    print('FALSE\n')
                # If is none of the five categories
                elif imRes[0][1] not in categ_names:
                    # Add 1 at value in cell with row same as folder name
                    whichRow = np.where(cm.index == wnids_list[i][0])
                    # and column as classified name, here 'Others'.
                    cm['Other'][whichRow[0][0]] += 1
                    print('OTHER\n')
                        
# Compute Acc and Rel in Confusion Matrix
# Rel
    for categ in cm.columns[:-1]:  # For every (column) category (and 'other')
        cm[categ][-1] = sum(cm[categ][:-1])  # Compute Total sum of column and write to last row [col][row]
        if categ != 'Other':  #  Only for real categories, compute Rel and replace value in last row
            # rel = diagonal element / last row element
            rel = (cm[categ][categ_names.index(categ)] / cm[categ][-1])
            # assign rel-value to last row
            cm[categ][-1] = round(rel, 2)
# Acc
    for categ in cm.index[:-1]: # For every (row) category (Here we dont have 'other')
        whichRow = np.where(cm.index == categ) # row is where index == categ name
        rowValues = cm.iloc[whichRow[0][0]][:-1]  # select the whole row
        # Acc = diagonal element / sum of row
        acc = cm[categ][categ_names.index(categ)] / sum(rowValues)
        # assign acc-value to last column 
        cm[cm.columns[-1]][categ_names.index(categ)] = round(acc, 2)
# f1-score
    for categ in cm.index[:-1]: # For every (row) category (Here we dont have 'other')
        # Rel ^-1 (inverse)
        whichRow = np.where(cm.index == 'Recall-User-Reliab') # row is where index == Rel
        relValues = cm.iloc[whichRow[0][0]][:-2]  # select the whole row - rowValues
        invRelValues = [1/x for x in relValues]
        # Acc ^-1 (inverse)
        accValues = cm['Precision-Producer-Accur'][:-1]
        invAccValues = [1/x for x in accValues]
        # f1-score
        f1 = (len(invRelValues)+len(invAccValues)) / (sum(invRelValues)+sum(invAccValues))
        # Assign f1 to bottom right element
        cm.iloc[whichRow[0][0]][-1] = round(f1, 2)
    return cm



if __name__ == "__main__": 
    # Change current working directory
    cwd = '/home/olyna/Documents/msc_Courses/EarthBigData/ask07'
    os.chdir(cwd)
    
    # Choose categories and corresponding wnids
    wnids_list = [('tiger_cat','n02123159'), ('Egyptian_cat','n02124075'), ('Persian_cat','n02123394'), ('tabby','n02123045'), ('Siamese_cat','n02123597')]

    # Create filesystem and save images (categ_names-list is needed for confusion matrix)
    categ_names = []
    for categ_name, wnid in wnids_list:
        save_images_from_wnid(cwd, categ_name, wnid, 12)
        categ_names.append(categ_name)
        
# ResNet50
    from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
    # Load the model architecture and the imagenet weights for the networks
    model = ResNet50(weights='imagenet') 
    #model.summary()
    #model.get_weights()[0]
    cmResNet50 = predictNevaluate(model, (224, 224), cwd, categ_names, wnids_list)
                  
# VGG16
    from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    model = VGG16(weights='imagenet')
    cmVGG16 = predictNevaluate(model, (224, 224), cwd, categ_names, wnids_list)
    
# MobileNet
    from tensorflow.python.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
    model = MobileNet(weights='imagenet')
    cmMobileNet = predictNevaluate(model, (224, 224), cwd, categ_names, wnids_list)
     
# Inception_V3
    from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
    model = InceptionV3(weights='imagenet')
    cmInception_V3 = predictNevaluate(model, (299, 299), cwd, categ_names, wnids_list)
 
# Xception
    from tensorflow.python.keras.applications.xception import Xception, preprocess_input, decode_predictions
    model = Xception(weights='imagenet')
    cmXception = predictNevaluate(model, (299, 299), cwd, categ_names, wnids_list)    
