import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
import cv2
from skimage.measure import compare_ssim, compare_psnr


def calc_diff(filename):
    pickle_in = open(filename, 'rb')
    x = pickle.load(pickle_in)
    pickle_in.close()
    #print(x)
    original = x["image"][0]
    adversarial = x["image"][len(x)-1]
    difference = adversarial - original
    return original, adversarial, difference


'''
original, adversarial, difference = calc_diff(0)
print(difference.shape)
#plt.imsave('Adversarial_snake.jpg', adversarial)
plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow((original / 255))
plt.axis('off')

#plt.subplot(1, 3, 2)
#plt.title('Adversarial')
plt.imshow((adversarial / 255))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
plt.imshow(difference)
plt.axis('off')

#png.from_array(difference, mode="L").save("diff.jpg")

plt.show()

cv2.imwrite('adversarial_snake.jpg', cv2.cvtColor(adversarial.astype('float32'),cv2.COLOR_RGB2BGR))

SSIM = compare_ssim(original, original_x, multichannel= True)
psnr = compare_psnr(original, original_x, data_range=255)

print("For 0th element of frame vs original image...")
print("Calculated SSIM is {} and frame SSIM is {}" .format(SSIM, x["ssim"][0]))
print("Calculated PSNR is {} and frame PSNR is {}" .format(psnr, x["psnr"][0]))

SSIM = compare_ssim(original, adversarial, multichannel= True)
psnr = compare_psnr(original, adversarial, data_range=255)

print("For last element of frame vs original image...")
print("Calculated SSIM is {} and frame SSIM is {}" .format(SSIM, x["ssim"][128]))
print("Calculated PSNR is {} and frame PSNR is {}" .format(psnr, x["psnr"][128]))
'''
