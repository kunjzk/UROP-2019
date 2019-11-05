import numpy as np
import cv2
from sparse_simba import main_sparse_simba, get_image
from CAM_keras import cam_keras
import os
import matplotlib.pyplot as plt


def compare_plots(avg_noise, original_heatmap_raw, plt_filename):
    noise_normalized = ((avg_noise+1)*255/2).astype('uint8')

    red_channel = noise_normalized[:,:,0]
    green_channel = noise_normalized[:,:,1]
    blue_channel = noise_normalized[:,:,2]
    plt.figure()
    plt.subplot(2,2,1)
    plt.title('red noise')
    plt.imshow(red_channel, cmap = 'gray')

    plt.subplot(2,2,2)
    plt.title('blue noise')
    plt.imshow(blue_channel, cmap = 'gray')

    plt.subplot(2,2,3)
    plt.title('green noise')
    plt.imshow(green_channel, cmap = 'gray')

    plt.subplot(2,2,4)
    plt.title('CAM')
    plt.imshow(original_heatmap_raw, cmap = 'gray')

    plt.savefig(plt_filename)


def compare_heatmaps(orig_heatmap, adv_heatmap, heatmap_figname, index):

    pathname_orig = "Results/{}/Original CAM.jpg".format(str(index))
    pathname_adv = "Results/{}/Adversarial CAM.jpg".format(str(index))

    orig_result = cv2.imread(pathname_orig)[:,:,::-1]
    adv_result = cv2.imread(pathname_adv)[:,:,::-1]
    plt.figure()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.subplot(2,2,1)
    plt.title('Original CAM + input')
    plt.imshow(orig_result)
    plt.subplot(2,2,2)
    plt.title('Original CAM')
    plt.imshow(orig_heatmap[:,:,::-1])
    plt.subplot(2,2,3)
    plt.title('Adversarial CAM + input')
    plt.imshow(adv_result)
    plt.subplot(2,2,4)
    plt.title('Adversarial CAM')
    plt.imshow(adv_heatmap[:,:,::-1])

    plt.savefig(heatmap_figname)



n_images = 10
n_iterations = 100
size = 8

for i in range(10):
    dirname = "Results/{}".format(str(i))
    os.mkdir(dirname)
    main_sparse_simba.generate_adversarial(size, i , n_iterations)

    #get the first instance of the original and adversarial image
    #to check predictions and CAMs
    original = np.load("sparse_simba/data/x_val_1000.npy")[i]
    pickle_filename = 'sparse_simba/pickles/local_resnet50_untargeted_SimBA_64_{}_img{}1.pickle'.format(str(size), str(i))
    _, adversarial, _ = get_image.calc_diff(pickle_filename)
    #print(original)
    plt.figure()
    plt.title('Orig image')
    plt.imshow(original.astype('uint8'))
    plt.axis('off')
    figname_o = dirname + '/original.png'
    plt.savefig(figname_o)

    # figure out how to return and handle appropriate values
    text_filename = dirname + "/results2.txt"
    orig_heatmap, orig_result, adv_heatmap, adv_result, original_heatmap_raw = cam_keras.get_CAM(original, adversarial, text_filename, i)

    heatmap_figname = dirname + "/heatmaps2.jpg"
    compare_heatmaps(orig_heatmap, adv_heatmap, heatmap_figname, i)
    #calculate average noise
    total_diff = np.zeros((224,224,3))
    for k in range(n_iterations):
        pickle_filename = 'sparse_simba/pickles/local_resnet50_untargeted_SimBA_64_{}_img{}{}.pickle'.format(str(size), str(i), str(k))
        _,_,diff = get_image.calc_diff(pickle_filename)
        total_diff += diff
    avg_noise = total_diff/n_iterations
    filename = dirname + '/avg_noise_s_{}_{}.npy'.format(str(size),str(n_iterations))
    np.save(filename, avg_noise)
    plt.figure()
    plt.title('Avg noise')
    plt.imshow(avg_noise)
    plt.axis('off')
    figname = dirname + '/avg_noise.png'
    plt.savefig(figname)

    plt_filename = dirname + "/noise_heatmap_comparison_{}.jpg".format(str(i))
    compare_plots(avg_noise, original_heatmap_raw, plt_filename)
