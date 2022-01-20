from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from skimage.feature import match_template
from skimage.measure import label

# ----------------------Image files------------------------------

# image = "FITS_images/P039+65_fullband-MFS-I-image-pb.fits"
image = "FITS_images/P045+69_fullband-MFS-I-image-pb.fits"
# image = "FITS_images/P048+65_fullband-MFS-I-image-pb.fits"

# ---------------------------Template samples for sample matching-------------

#template_image_045 = 'FITS_images/Templates/P045_P048_template_1.fits'
template_image_045 = 'FITS_images/Templates/P045_P048_template_3.fits'

template_file = fits.open(template_image_045)
template_data = template_file[0].data

# plt.subplot(121)
# plt.title('before')
# plt.imshow(template_data)

# remove nan values
nn = np.argwhere(~np.isnan(template_data))
template_data = template_data[np.min(nn[0]):nn[-1][0],nn[0][-1]:np.max(nn[:-1])]
# plt.subplot(122)
# plt.title('after')
#plt.imshow(template_data)
#plt.show()

# --------------------------------------Image data------------------------
image_file = fits.open(image)

# image data
image_data = image_file[0].data

# image header
image_header = image_file[0].header
# ++++++++++++++++++++++++initialize wcs++++++++++++++++++++++++++++++++++++++++++++++++++

wcs = WCS(header=image_header, fix=True)[0, 0, :, :]

# /++++++++++++++++++++++++initialize wcs++++++++++++++++++++++++++++++++++++++++++++++++++

# ++++++++++++++++++++++++image header++++++++++++++++++++++++++++++++++++++++++++++++++
# print('\n Header:')
# print(repr(image_file[0].header))

# print('\n----------------------------------')
# print('\nCTYPE 3: ',image_file[0].header['CTYPE3'])
# print('\nNAXIS: ',image_file[0].header['NAXIS'])
# print('\nNAXIS 1: ',image_file[0].header['NAXIS1'])
# print('Telescope: ', image_file[0].header['TELESCOP'])
# /++++++++++++++++++++++++image header++++++++++++++++++++++++++++++++++++++++++++++++++
top=0.956
bottom=0.1
left=0.068
right=0.988
hspace=0.5
wspace=0.2
# plot

fig = plt.figure(figsize=(16, 16), dpi=100)
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
ax = fig.add_subplot(121, projection=wcs)
ax2 = fig.add_subplot(122)
# plot in log scale
# data_abs = np.abs(image_data[0, 0, :, :])
# data_log = np.log10(data_abs) * 10

ax.set_title('Original image')
ax.set_xlabel('Right ascension [h:m:s]')
ax.set_ylabel('Declination [deg]')
im = ax.imshow(image_data[0, 0, :, :], cmap='gray')
#add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('[Jy / Beam]')

# histogram
NBINS = 1000

y1, x1, _ = ax2.hist(image_data[0, 0, :, :].flatten(), NBINS)
ax2.set_yscale('log')
ax2.set_title('Original histogram')
ax2.set_xlabel('Pixel intensity [Jy/Beam]')
ax2.set_ylabel('Pixel Count')

# -------------------------intervals using max and std-------------------------------------

x1_max = x1[np.where(y1 == y1.max())]
std = np.std(image_data)

interval_dict = {}
for i in range(1, 10):
    interval_dict["interval{0}".format(i)] = np.clip(image_data[0, 0, :, :], x1_max - i * std, x1_max + i * std)

intervals = list(interval_dict.values())

# ----------------------------------------------- plot all interval images-------------------------------------------------
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 16), subplot_kw={'projection': wcs})
index = 0
for i in range(0,3):
    for j in range(0,3):
        im = ax[i][j].imshow(intervals[index])
        ax[i][j].set_title('Image +/- '+str(index+1)+' intervals')
        ax[i][j].set_xlabel('Right ascension [h:m:s]')
        ax[i][j].set_ylabel('Declination [deg]')
        cbar = plt.colorbar(im, ax=ax[i][j])
        cbar.set_label('[Jy / Beam]')
        index += 1
top=0.956
bottom=0.1
left=0.068
right=0.97
hspace=0.5
wspace=0.1
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)

#histograms
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))

index = 0
for i in range(0,3):
    for j in range(0,3):
        ax[i][j].hist(intervals[index].flatten(), NBINS)
        ax[i][j].set_title('Histogram in '+str(index+1)+' intervals')
        ax[i][j].set_xlabel('Pixel intensity [Jy/Beam]')
        ax[i][j].set_ylabel('Pixel Count')
        ax[i][j].set_yscale('log')
        index += 1

top=0.956
bottom=0.1
left=0.068
right=0.988
hspace=0.5
wspace=0.2
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
plt.tight_layout()

# ---------------------------------------Template matching-------------------------------

fig = plt.figure(figsize=(16, 16), dpi=100)
ax = fig.add_subplot(121, projection=wcs)
ax2 = fig.add_subplot(122, projection=wcs)

image_for_template_matching = intervals[6]

ax.set_title('Image')
ax.set_xlabel('Right ascension [h:m:s]')
ax.set_ylabel('Declination [deg]')
ax.imshow(image_for_template_matching)
ax2.set_title('Template')
ax2.set_xlabel('Right ascension [h:m:s]')
ax2.set_ylabel('Declination [deg]')
ax2.imshow(template_data)
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)


def matchTemplate(im, template, Th):  
    #find correlation coefficients
    res = match_template(im, template, pad_input=True)
    #find goodd matching pixels
    BW = res >= Th
    #get each pixel
    L, num = label(BW, return_num=True)
    coords = np.zeros((num, 2))
    for i in range(0, num):
        #split each binary
        connComp = L == i
        gammaC = deepcopy(res)
        gammaC[connComp == 0] = 0
        #find coordinates for pixel which`s value ir greatest
        ind = np.unravel_index(np.argmax(gammaC, axis=None), gammaC.shape)
        #save coords
        coords[i - 1, :] = ind
    return coords

coords=matchTemplate(image_for_template_matching, template_data, 0.2)
print('image for template matching shape ', image_for_template_matching.shape)
print('template shape ', template_data.shape)
print('coords shape ', coords.shape)
print('coords ', coords)
fig = plt.figure(figsize=(16, 16), dpi=100)
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
ax = fig.add_subplot(111, projection=wcs)

ax.set_title('Matched objects')
ax.set_xlabel('Right ascension [h:m:s]')
ax.set_ylabel('Declination [deg]')
im = ax.imshow(image_for_template_matching)
#add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('[Jy / Beam]')
plt.scatter(coords[:,1], coords[:,0], c='r')
plt.show()

# -----------------------Pixel to world-------------------------------------------------------------
# get neccessary data for coordinate transformations
w = WCS(image_header, fix=True)
# print('w: ', w)

# pixel to world
pixel_to_world = w.pixel_to_world(coords, coords, coords, coords)
print('\nPixel to world: ', pixel_to_world)
