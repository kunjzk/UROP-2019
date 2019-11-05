"""
Author: NVS Abhilash

Keras implementation of Class Activation Mappings: http://cnnlocalization.csail.mit.edu/
This script is based on pytorch version of the same: https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py

"""

import numpy as np
import cv2
import io
import requests
from PIL import Image
import shutil
import os
import pickle

# Using Keras implementation from tensorflow
from tensorflow.python.keras import applications
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications.resnet50 import preprocess_input

# Function to generate Class Activation Mapping
def returnCAM(feature_conv, weight_softmax, class_idx):
    HEIGHT = 224
    WIDTH = 224
    size_upsample = (WIDTH, HEIGHT)
    # Keras default is channels last, hence nc is in last
    bz, h, w, nc = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = np.dot(weight_softmax[:, idx], np.transpose(feature_conv.reshape(h*w, nc)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam


def get_CAM(orig_image_in, adv_image_in, filepath, index):
    K.clear_session()
    LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

    HEIGHT = 224
    WIDTH = 224

    model = applications.ResNet50(include_top=True)
    #model.summary()
    finalconv_name = 'activation_48'
    fianlconv = model.get_layer(finalconv_name)
    weight_softmax = model.layers[-1].get_weights()[0]

    orig_image = preprocess_input(orig_image_in)
    adv_image = preprocess_input(adv_image_in)

    probs_extractor = K.function([model.input], [model.output])

    # This is how we get intermediate layer output in Keras (this returns a callable)
    features_conv_extractor = K.function([model.input], [fianlconv.output])

    classes = {int(key):value for (key, value) in requests.get(LABELS_URL).json().items()}

    # Getting final layer output
    probs = probs_extractor([np.expand_dims(orig_image, 0)])[0]
    # Getting output of last conv layer
    features_blob = features_conv_extractor([np.expand_dims(orig_image, 0)])[0]
    features_blobs = []
    features_blobs.append(features_blob)
    idx = np.argsort(probs)
    probs = np.sort(probs)

    with open(filepath, "w") as text_file:
        #for Original Image
        text_file.write('For original image:')
        # Reverse loop to print highest prob first.
        for i in range(-1, -6, -1):
            text_file.write('{:.3f} -> {}'.format(probs[0, i], classes[idx[0, i]]))

        text_file.write("\n")
        text_file.write('top1 prediction: {}'.format(classes[idx[0, -1]]))

        text_file.write("\n\n\n")

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
    original_heatmap_raw = CAMs[0]
    height, width, _ = orig_image.shape
    original_heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    original_result = original_heatmap * 0.3 + orig_image * 0.5
    cv2.imwrite('Results/{}/Original CAM.jpg'.format(str(index)), original_result)


    # Getting final layer output
    probs = probs_extractor([np.expand_dims(adv_image, 0)])[0]
    # Getting output of last conv layer
    features_blob = features_conv_extractor([np.expand_dims(adv_image, 0)])[0]
    features_blobs = []
    features_blobs.append(features_blob)
    idx = np.argsort(probs)
    probs = np.sort(probs)

    with open(filepath, "a") as text_file:
        #for Adversarial Image
        text_file.write('For adversarial image:')
        # Reverse loop to print highest prob first.
        for i in range(-1, -6, -1):
            text_file.write('{:.3f} -> {}'.format(probs[0, i], classes[idx[0, i]]))

        text_file.write("\n")
        text_file.write('top1 prediction: {}'.format(classes[idx[0, -1]]))

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
    height, width, _ = adv_image.shape
    adv_heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    adv_result = adv_heatmap * 0.3 + adv_image * 0.5
    cv2.imwrite('Results/{}/Adversarial CAM.jpg'.format(str(index)), adv_result)

    return original_heatmap, original_result, adv_heatmap, adv_result, original_heatmap_raw




'''
def generate_CAM(orig_image_in, adv_image_in):
    LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

    HEIGHT = 224
    WIDTH = 224

    model = applications.ResNet50(include_top=True)

    finalconv_name = 'activation_97'

    # Get the layer of the last conv layer
    fianlconv = model.get_layer(finalconv_name)

    # Get the weights matrix of the last layer
    weight_softmax = model.layers[-1].get_weights()[0]

    orig_image = preprocess_input(orig_image_in)
    adv_image = preprocess_input(adv_image_in)

    probs_extractor = K.function([model.input], [model.output])

    # This is how we get intermediate layer output in Keras (this returns a callable)
    features_conv_extractor = K.function([model.input], [fianlconv.output])

    classes = {int(key):value for (key, value) in requests.get(LABELS_URL).json().items()}


    # Getting final layer output
    probs = probs_extractor([np.expand_dims(orig_image, 0)])[0]
    # Getting output of last conv layer
    features_blob = features_conv_extractor([np.expand_dims(orig_image_in, 0)])[0]
    features_blobs = []
    features_blobs.append(features_blob)
    idx = np.argsort(probs)
    probs = np.sort(probs)
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
    original_heatmap_raw = CAMs[0]
    #print('output CAM.jpg for top1 prediction: {}'.format(classes[idx[0, -1]]))
    #img = cv2.imread('test.jpg')
    height, width, _ = orig_image_in.shape
    original_heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    original_result = original_heatmap * 0.3 + orig_image_in * 0.5
    cv2.imwrite('Results/0/CAM.jpg', original_result)


    ##-----FOR ADVERSARIAL----

    probs = probs_extractor([np.expand_dims(adv_image, 0)])[0]
    # Getting output of last conv layer
    features_blob = features_conv_extractor([np.expand_dims(adv_image, 0)])[0]
    features_blobs = []
    features_blobs.append(features_blob)
    idx = np.argsort(probs)
    probs = np.sort(probs)
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
    height, width, _ = adv_image.shape
    adv_heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    adv_result = adv_heatmap * 0.3 + adv_image * 0.5
    return original_heatmap, original_result, adv_heatmap, adv_result, original_heatmap_raw
'''
