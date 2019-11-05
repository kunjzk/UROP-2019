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
#import get_image
import matplotlib.pyplot as plt

import sparse_simba.utils as utils

def setup_local_model():
    #sets up local ResNet50 model, to use for local testing
    keras.backend.set_learning_phase(0)
    kmodel = keras.applications.resnet50.ResNet50(weights='imagenet')
    #kmodel.summary()
    preprocessing = (np.array([104, 116, 123]), 1)
    model = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing, predicts='logits')
    #print(dir(model))
    return model

def generate_adversarial(size, i, j_range):
    log_every_n_steps = 100 #log progress to console every n steps
    query_limit = 5000 #set to None for queryless setting
    epsilon = 64
    size = 8

    setting = 'untargeted'
    target_system = 'local_resnet50'

    ################## -- START OF ATTACK -- #######################

    x_val = np.load("sparse_simba/data/x_val_1000.npy") #loads 1000 instances of the ImageNet validation set
    y_val = np.load("sparse_simba/data/y_val_1000.npy") #loads labels of the 1000 instances of the ImageNet validation set

    print('loading untargeted and targeted splits...')
    local_untargeted_split = utils.pickle_load('sparse_simba/data/untargeted_split.pickle') #the indices of the random split of the images we will be testing
    local_targeted_split = utils.pickle_load('sparse_simba/data/targeted_split.pickle') #the indices of the random split of the images we will be testing
    #api_classifiers_untargeted_split = utils.pickle_load('data/online_api_classifiers_untargeted_split.pickle')
    #api_classifiers_targeted_split = utils.pickle_load('data/online_api_classifiers_targeted_split.pickle')

    print('starting simba attack...')
    print('epsilon: {} ({:.2%})'.format(epsilon, epsilon/255))
    print('size: {}, {}, max directions: {} ({:.2%})'.format(size, size, 224*224*3/size/size, 1/size/size))

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
        split = targeted_split.T

    #print(split)

    for j in range(j_range): #calculate j noise distros
#        for i in range(i_range): #loop through all images in the split
        df_filename = 'sparse_simba/pickles/{}_{}_SimBA_{}_{}_img{}{}.pickle'.format(str(target_system), str(setting), str(epsilon), str(size), str(i), str(j))
        file_exists = os.path.isfile(df_filename)
        if file_exists:
            print('file for image {} already exists'.format(i))
        else:
            if setting == 'targeted':
                #unpack targeted labels
                target_class = i[1]
                i = int(i[0])
            #print(i)
            original_image = x_val[i] #img must be bgr
            original_class = y_val[i]
            start = time.time()

            adv, total_calls, info_df = utils.run_sparse_simba(original_image, size=size, epsilon=epsilon, setting=setting,
                                                query_limit=query_limit, target_system=target_system,
                                                target_class=target_class, local_model=local_model,
                                                log_every_n_steps=log_every_n_steps) #set size=1 for original SimBA

            print('------ ATTACK {} -------'.format(j))
            print('total time taken: {}s'.format(time.time()-start))
            print('total queries made: {}'.format(total_calls))

            #improve these last 3 before AWS
            utils.pickle_save(info_df, df_filename)
            # img_filename = df_filename[:-7] + '.png'
            # save_adv_details_simba(original_image, adv, total_calls, aws_model, img_filename)

    keras.backend.clear_session()
'''
total_dif  = np.zeros((224,224,3))
for k in range(1000):
    _,_,diff = get_image.calc_diff(k)
    total_diff += diff

avg_noise = total_diff/1000
np.save('avg_noise_1000_s_2.npy',avg_noise)

plt.figure()
plt.title('Avg noise for 1000 iterations, s=2')
plt.imshow(avg_noise)
plt.axis('off')
plt.imsave('avg_noise_s_2.png', avg_noise)
plt.show()
'''
