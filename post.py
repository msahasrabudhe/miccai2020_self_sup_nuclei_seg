import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure
from skimage import morphology

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, blob_log

import matplotlib.pylab as plt

def post(img):
    img_post = img
    
    img_post = ndimage.binary_opening(img_post, structure=morphology.disk(2)).astype(np.int)
    img_post = ndimage.binary_closing(img_post, structure=morphology.disk(2)).astype(np.int)
    img_post = ndimage.binary_opening(img_post, structure=morphology.disk(1)).astype(np.int)
    img_post = ndimage.binary_closing(img_post, structure=morphology.disk(1)).astype(np.int)

#    Image.fromarray(255*img_post[:512,:512].astype('uint8')).save('postfig-morph.png')

    distance = ndimage.morphology.distance_transform_edt(img_post)
    smoothed_distance = ndimage.gaussian_filter(distance, sigma=1)


    local_maxi = peak_local_max(smoothed_distance, indices=False, footprint=morphology.disk(7), labels=img_post) # morphology.disk(10)
    markers = ndi.label(local_maxi)[0]

    # saving distance image with peaks
#    toplt = np.array(Image.fromarray(255*(smoothed_distance[:512,:512]/np.max(smoothed_distance[:512,:512]))).convert('RGB'))
#    buffedmaxi = ndimage.binary_dilation(local_maxi, structure=morphology.disk(2)).astype(np.int)
#    toplt[buffedmaxi[:512,:512]==1] = np.array([255,0,0])
#    Image.fromarray(toplt).save('postfig-dist.png')

    img_post = watershed(-smoothed_distance, markers, mask=img_post)

    return img_post


def watershed_post(img):
    distance = ndimage.morphology.distance_transform_edt(img.astype(np.int))
    smoothed_distance = ndimage.gaussian_filter(distance, sigma=1)

    local_maxi = peak_local_max(smoothed_distance, indices=False, footprint=morphology.disk(3), labels=img)
    markers = ndi.label(local_maxi)[0]

    img_post = watershed(-smoothed_distance, markers, mask=img)
    return img_post
