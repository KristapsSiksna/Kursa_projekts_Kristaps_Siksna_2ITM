# -*- coding: utf-8 -*-
"""
=======================================
Read and plot an image from a FITS file
=======================================

This example opens an image stored in a FITS file and displays it to the screen.

This example uses `astropy.utils.data` to download the file, `astropy.io.fits` to open
the file, and `matplotlib.pyplot` to display the image.


*By: Lia R. Corrales, Adrian Price-Whelan, Kelle Cruz*

*License: BSD*


"""

##############################################################################
# Set up matplotlib and use a nicer set of plot parameters

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

##############################################################################
# Download the example FITS files used by this example:

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

image_file = get_pkg_data_filename('tutorials/FITS-images/HorseHead.fits')

##############################################################################
# Use `astropy.io.fits.info()` to display the structure of the file:

fits.info(image_file)

##############################################################################
# Generally the image information is located in the Primary HDU, also known
# as extension 0. Here, we use `astropy.io.fits.getdata()` to read the image
# data from this first extension using the keyword argument ``ext=0``:

image_data = fits.getdata(image_file, ext=0)

##############################################################################
# The data is now stored as a 2D numpy array. Print the dimensions using the
# shape attribute:

print(image_data.shape)
print("Image data: ", image_data)

print('Min:', np.min(image_data))
print('Max:', np.max(image_data))
print('Mean:', np.mean(image_data))
print('Stdev:', np.std(image_data))

##############################################################################
# Display the image data:

plt.figure()
plt.imshow(image_data, cmap='gray')
plt.colorbar()
plt.show()

#histogram
print(type(image_data.flatten()))
NBINS = 1000
histogram = plt.hist(image_data.flatten(), NBINS)
plt.show()


# numpy clip to make histogram more effective
# make [-0.1, 0.1] interval
# a = np.clip(image_data[0, 0, :, :], -0.1, 0.1)
# print('Stdev a [-0.1, 0.1] :', np.std(a))
#
# # image [-0.1, 0.1] interval
# plt.figure(figsize=(12, 8))
# plt.subplot(121)
# plt.title('Image a in [-0.1 0.1] interval')
# plt.imshow(a, cmap='cool')
# plt.colorbar()
#
# # histogram in [-0.1, 0.1] interval
# plt.subplot(122)
# plt.title('Histogram a in [-0.1 0.1] interval')
# y2, x2, _ = plt.hist(a.flatten(), NBINS)
# plt.tight_layout()
#
# # max y and corresponding x
# x2_max = x2[np.where(y2 == y2.max())]
# print('x2 max: ', x2_max)
#
# #make [-0.01, 0.01] interval
# b = np.clip(image_data[0, 0, :, :], -0.01, 0.01)
# print('\nStdev b [-0.01, 0.01]:', np.std(b))
#
# # image [-0.01, 0.01] interval
# plt.figure(figsize=(12, 8))
# plt.subplot(121)
# plt.title('Image b in [-0.01 0.01] interval')
# plt.imshow(b, cmap='cool')
# plt.colorbar()
#
# #histogram in [-0.01, 0.01] interval
# plt.subplot(122)
# plt.title('Histogram b in [-0.01 0.01] interval')
# y3, x3, _ = plt.hist(b.flatten(), NBINS)
# plt.tight_layout()
# plt.show()
#
# # max y and corresponding x
# x3_max = x3[np.where(y3 == y3.max())]
# print('x3 max: ', x3_max)
# print('a max', a.max())
# print('b max', b.max())
# print('interval 1 max', interval1.max())
# print('interval 2 max', interval2.max())
# print('interval 3 max', interval3.max())

#------

# interval1 = np.clip(image_data[0, 0, :, :], x1_max - std, x1_max + std)
# print('1 * std: ', 1 * std)
#
# interval2 = np.clip(image_data[0, 0, :, :], x1_max - 2 * std, x1_max + 2 * std)
# print('2 * std: ', 2 * std)
#
# interval3 = np.clip(image_data[0, 0, :, :], x1_max - 3 * std, x1_max + 3 * std)
# print('3 * std: ', 3 * std)
#
# # plot interval
# plt.figure(figsize=(15, 8))
#
# # 1st interval
# # plt.subplot(231)
# # plt.title('Image in [max-std,max+std] interval LOG NORM')
# # interval1_abs = np.abs(interval1)
# # interval1_log = np.log10(interval1_abs)
# # plt.imshow(interval1_log, cmap='cool')
# # plt.colorbar()
#
# plt.subplot(231)
# plt.title('Image in [max-std,max+std] interval')
# plt.imshow(interval1, cmap='cool')
# plt.colorbar()
#
# # 2nd interval
# plt.subplot(232)
# plt.title('Image in [max-2*std,max+2*std] interval')
# plt.imshow(interval2, cmap='cool')
# plt.colorbar()
#
# # 3rd interval
# plt.subplot(233)
# plt.title('Image in [max-3*std,max+3*std] interval')
# plt.imshow(interval3, cmap='cool')
# plt.colorbar()
# # plt.tight_layout()
#
# # histograms
# plt.subplot(234)
# plt.title('Histogram in [max-std,max+std] interval LOG scale')
# plt.hist(interval1.flatten(), NBINS)
# plt.yscale('log')
#
# plt.subplot(235)
# plt.title('Histogram in [max-2*std,max+2*std] interval LOG scale')
# plt.hist(interval2.flatten(), NBINS)
# plt.yscale('log')
#
# plt.subplot(236)
# plt.title('Histogram in [max-3*std,max+3*std] interval LOG scale')
# plt.hist(interval3.flatten(), NBINS)
# plt.yscale('log')
# plt.tight_layout()
# plt.show()
#
# print('\ninterval 1 max', interval1.max())
# print('interval 2 max', interval2.max())
# print('interval 3 max', interval3.max())
#
# exit()
#
# print('interval 1 shape: ', interval1.shape)
# print('template image shape: ', template_data.shape)