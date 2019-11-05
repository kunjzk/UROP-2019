import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import keras
import cv2
import boto3
from skimage.measure import compare_ssim, compare_psnr
from foolbox.models import KerasModel
import vis.visualization as v
from tensorflow.keras import layers
from CAM_keras import cam_keras
#import get_image
import matplotlib.pyplot as plt
import sparse_simba.utils as utils
import requests
from sparse_simba import get_image

from tensorflow.python.keras import applications
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications.resnet50 import preprocess_input


def setup_local_model():
    #sets up local ResNet50 model, to use for local testing
    keras.backend.set_learning_phase(0)
    kmodel = keras.applications.resnet50.ResNet50(weights='imagenet')
    preprocessing = (np.array([104, 116, 123]), 1)
    model = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing, predicts='logits')
    return model


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


def get_CAM(orig_image_in, target_image_in, adv_image_in):
    K.clear_session()

    HEIGHT = 224
    WIDTH = 224

    model = applications.ResNet50(include_top=True)
    #model.summary()
    finalconv_name = 'activation_48'
    fianlconv = model.get_layer(finalconv_name)
    weight_softmax = model.layers[-1].get_weights()[0]

    orig_image_copy = orig_image_in

    orig_image = preprocess_input(orig_image_in)
    target_image = preprocess_input(target_image_in)
    adv_image = preprocess_input(adv_image_in)

    probs_extractor = K.function([model.input], [model.output])

    # This is how we get intermediate layer output in Keras (this returns a callable)
    features_conv_extractor = K.function([model.input], [fianlconv.output])

    # Getting final layer output
    probs = probs_extractor([np.expand_dims(orig_image, 0)])[0]
    # Getting output of last conv layer
    features_blob = features_conv_extractor([np.expand_dims(orig_image, 0)])[0]
    features_blobs = []
    features_blobs.append(features_blob)
    idx = np.argsort(probs)
    probs = np.sort(probs)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
    original_heatmap_raw = CAMs[0]
    height, width, _ = orig_image.shape
    original_heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    original_result = original_heatmap * 0.3 + orig_image_copy * 0.5



    # Getting final layer output
    probs = probs_extractor([np.expand_dims(adv_image, 0)])[0]
    # Getting output of last conv layer
    features_blob = features_conv_extractor([np.expand_dims(adv_image, 0)])[0]
    features_blobs = []
    features_blobs.append(features_blob)
    idx = np.argsort(probs)
    probs = np.sort(probs)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
    adv_heatmap_raw = CAMs[0]
    height, width, _ = adv_image.shape
    adv_heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    adv_result = adv_heatmap * 0.3 + adv_image_in * 0.5



    # Getting final layer output
    probs = probs_extractor([np.expand_dims(target_image, 0)])[0]
    # Getting output of last conv layer
    features_blob = features_conv_extractor([np.expand_dims(target_image, 0)])[0]
    features_blobs = []
    features_blobs.append(features_blob)
    idx = np.argsort(probs)
    probs = np.sort(probs)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
    target_heatmap_raw = CAMs[0]
    height, width, _ = target_image.shape
    target_heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    target_result = target_heatmap * 0.3 + target_image_in * 0.5


    return original_heatmap, original_result, target_heatmap, target_result, adv_heatmap, adv_result, original_heatmap_raw, adv_heatmap_raw, target_heatmap_raw




def plot_heatmaps(orig_heatmap_in, original_composite, target_heatmap_in, target_composite, adv_heatmap_in, adv_composite, count, original_class, target_class, adversarial_class, dirname):

    plt.figure()
    plt.subplots_adjust(wspace=1.5, hspace=0.5)
    plt.subplot(2,3,1)
    plt.title('{}'.format(str(original_class)))
    plt.imshow(original_composite[:,:,::-1]/256)
    plt.subplot(2,3,4)
    plt.title('Original CAM')
    plt.imshow(orig_heatmap_in[:,:,::-1])
    plt.subplot(2,3,2)
    plt.title('{}'.format(str(target_class)))
    plt.imshow(target_composite[:,:,::-1]/256)
    plt.subplot(2,3,5)
    plt.title('Target CAM')
    plt.imshow(target_heatmap_in[:,:,::-1])
    plt.subplot(2,3,3)
    plt.title('{}'.format(str(adversarial_class)))
    plt.imshow(adv_composite[:,:,::-1]/256)
    plt.subplot(2,3,6)
    plt.title('Adversarial CAM')
    plt.imshow(adv_heatmap_in[:,:,::-1])

    plt.tight_layout()

    heatmap_figname = dirname + '/taegeted_heatmaps.jpg'.format(str(count))
    plt.savefig(heatmap_figname)


def compare_heatmaps(noise, original_heatmap, target_heatmap, adv_heatmap, count, dirname):

    red_channel = noise[:,:,0]
    green_channel = noise[:,:,1]
    blue_channel = noise[:,:,2]
    plt.figure()
    plt.subplot(2,3,1)
    plt.title('red noise')
    plt.imshow(red_channel, cmap = 'gray')

    plt.subplot(2,3,2)
    plt.title('blue noise')
    plt.imshow(blue_channel, cmap = 'gray')

    plt.subplot(2,3,3)
    plt.title('green noise')
    plt.imshow(green_channel, cmap = 'gray')

    plt.subplot(2,3,4)
    plt.title('Original image CAM')
    plt.imshow(original_heatmap_raw, cmap = 'gray')

    plt.subplot(2,3,5)
    plt.title('Target image CAM')
    plt.imshow(target_heatmap, cmap = 'gray')

    plt.subplot(2,3,6)
    plt.title('Adversarial image CAM')
    plt.imshow(adv_heatmap, cmap = 'gray')

    plt.tight_layout()

    plt_filename = dirname + "/targeted_heatmap_comparison.jpg".format(str(count))

    plt.savefig(plt_filename)





#"Main" function


log_every_n_steps = 100 #log progress to console every n steps
query_limit = 5000 #set to None for queryless setting
epsilon = 64
size = 8

setting = 'targeted'
target_system = 'local_resnet50'

################## -- START OF ATTACK -- #######################

x_val = np.load("sparse_simba/data/x_val_1000.npy") #loads 1000 instances of the ImageNet validation set
y_val = np.load("sparse_simba/data/y_val_1000.npy") #loads labels of the 1000 instances of the ImageNet validation set

print('loading untargeted and targeted splits...')
local_untargeted_split = utils.pickle_load('sparse_simba/data/untargeted_split.pickle') #the indices of the random split of the images we will be testing
local_targeted_split = utils.pickle_load('sparse_simba/data/targeted_split.pickle') #the indices of the random split of the images we will be testing


print('starting simba attack...')
print('epsilon: {} ({:.2%})'.format(epsilon, epsilon/255))
print('size: {}, {}, max directions: {} ({:.2%})'.format(size, size, 224*224*3/size/size, 1/size/size))


# for getting labels
synset_to_keras_idx = {}
keras_idx_to_name = {}
f = open("sparse_simba/data/synset_words.txt","r")
idx = 0
for line in f:
    parts = line.split(" ")
    synset_to_keras_idx[parts[0]] = idx
    keras_idx_to_name[idx] = " ".join(parts[1:])
    idx += 1
f.close()


if target_system == 'local_resnet50':
    #local model testing
    local_model = setup_local_model()
    untargeted_split = local_untargeted_split
    targeted_split = local_targeted_split
elif target_system == 'AWS' or target_system == 'GCV':
    local_model = None
    untargeted_split = api_classifiers_untargeted_split
    targeted_split = api_classifiers_targeted_split
else:
    raise Exception('target_system should be set to "local_resnet50", or "AWS" or "GCV"')

if setting == 'untargeted':
    target_class = None
    split = untargeted_split
elif setting == 'targeted':
    print(targeted_split)
    split = targeted_split

print(split)

is_adv = 0


for count, i in enumerate(split): #loop through all images in the split
    dirname = "Results/Targeted/{}".format(str(count))
    print(i)
    #if not os.path.exists(dirname):
        #os.mkdir(dirname)
    target_class = int(i[1])
    numpy_index = int(i[0])
    original_image = x_val[numpy_index]
    original_class = y_val[numpy_index]


    df_filename = 'sparse_simba/pickles_targeted/{}_{}_SimBA_{}_{}_img{}.pickle'.format(str(target_system), str(setting), str(epsilon), str(size), str(i))
    file_exists = os.path.isfile(df_filename)

    if file_exists:
        print('file for image {} already exists'.format(i))
        #load adv
        _, adv, noise = get_image.calc_diff(df_filename)
        #print(adv)
    else:
        start = time.time()

        adv, total_calls, info_df, is_adv = utils.run_sparse_simba(original_image, size=size, epsilon=epsilon, setting=setting,
                                            query_limit=query_limit, target_system=target_system,
                                            target_class=target_class, local_model=local_model,
                                            log_every_n_steps=log_every_n_steps) #set size=1 for original SimBA

        print('------ ATTACK {} -------'.format(count))
        print('total time taken: {}s'.format(time.time()-start))
        print('total queries made: {}'.format(total_calls))

        utils.pickle_save(info_df, df_filename)

        _, _, noise = get_image.calc_diff(df_filename)


    #if is_adv==1:
        #want to obtain all 3 images and classify them
    original_class = keras_idx_to_name[y_val[numpy_index]].split(',')[0]
    target_class_name = keras_idx_to_name[y_val[target_class]].split(',')[0]
    #print(adv)
    adversarial_class = keras_idx_to_name[y_val[int(utils.return_top_1_label(adv, local_model))]].split(',')[0]

    plt.figure()
    plt.title('adv')
    plt.imshow(adv/256)
    adv_filename = dirname + "/adv.jpg"
    plt.savefig(adv_filename)

    orig_copy = original_image
    target_copy = x_val[target_class]
    adv_copy = adv
    orig_CAM, orig_composite, target_CAM, target_composite, adv_CAM, adv_composite, original_heatmap_raw, adv_heatmap_raw, target_heatmap_raw = get_CAM(orig_copy, target_copy, adv_copy)

    np.save("original_heatmap.npy", original_heatmap_raw)
    np.save("target_heatmap.npy", target_heatmap_raw)
    np.save("adv_heatmap.npy", adv_heatmap_raw)

    plot_heatmaps(orig_CAM, orig_composite, target_CAM, target_composite, adv_CAM, adv_composite, count, original_class, target_class_name, adversarial_class, dirname)

    plt.figure()
    plt.title('Noise')
    plt.imshow(noise)
    noise_filename = dirname + '/targeted_noise.jpg'.format(str(count))
    plt.savefig(noise_filename)

    compare_heatmaps(noise, original_heatmap_raw, target_heatmap_raw, adv_heatmap_raw, count, dirname)
