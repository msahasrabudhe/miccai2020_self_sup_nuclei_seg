import glob
import post
import numpy as np
from scipy import ndimage
from skimage import morphology
import matplotlib.pylab as plt
from PIL import Image
import scipy.io as sio
from scipy import spatial
from matplotlib.pyplot import Polygon
from matplotlib.collections import PatchCollection
import SimpleITK as sitk
import skimage

def aji(gt_map,predicted_map):
    gt_list = np.unique(gt_map)
    gt_list = gt_list[1:]
    ngt = len(gt_list)
    pr_list = np.unique(predicted_map)
    pr_list = pr_list[1:]
    pr_list =  np.asarray([pr_list, np.zeros(len(pr_list))]).T
    npredicted = len(pr_list)

    overall_correct_count = 0
    union_pixel_count = 0

    i = len(gt_list)-1
    while len(gt_list)>0:
        gt = 1*(gt_map == gt_list[i])
        predicted_match = gt*predicted_map

        if np.sum(predicted_match) == 0:
            union_pixel_count += np.sum(gt)
            gt_list = np.delete(gt_list,i)
            i = len(gt_list)-1
        else:
            predicted_nuc_index = np.unique(predicted_match)
            predicted_nuc_index = predicted_nuc_index[1:]
            JI = 0
            for j in np.unique(predicted_nuc_index):
                matched = 1*(predicted_map == j)
                nJI =   np.sum(np.logical_and(gt,matched))/np.sum(np.logical_or(gt,matched))
                if nJI > JI:
                    best_match = j
                    JI = nJI

            predicted_nuclei = 1*(predicted_map == best_match)
            overall_correct_count += np.sum(np.sum(np.logical_and(gt,predicted_nuclei)))
            union_pixel_count += np.sum(np.sum(np.logical_or(gt,predicted_nuclei)))
            gt_list = np.delete(gt_list,i)
            i = len(gt_list)-1

            index = np.where(pr_list==best_match)[0]
            pr_list[index,1] = pr_list[index,1] + 1

    unused_nuclei_list  = np.where(pr_list[:,1]==0)[0]
    for k in unused_nuclei_list:
        unused_nuclei = 1*(predicted_map == pr_list[k,0])
        union_pixel_count = union_pixel_count + np.sum(unused_nuclei)

    aji = overall_correct_count/union_pixel_count
    return aji

def hausdorff(gt_map,predicted_map):
    gt_list = np.unique(gt_map)
    gt_list = gt_list[1:]
    ngt = len(gt_list)
    pr_list = np.unique(predicted_map)
    pr_list = pr_list[1:]
    pr_list =  np.asarray([pr_list, np.zeros(len(pr_list))]).T
    npredicted = len(pr_list)

    accumulated_HD = []
    x, y = np.meshgrid(np.arange(gt_map.shape[0]), np.arange(gt_map.shape[1]))
    for i in range(len(gt_list)):
        gt = 1*(gt_map == gt_list[i])
        predicted_match = gt*predicted_map
        hd = []
        for j in np.unique(predicted_match)[1:]:
            u = [x[gt>0], y[gt>0]]
            v = [x[predicted_match==j], y[predicted_match==j]]
            xxx = spatial.distance.directed_hausdorff(np.asarray(u).T,np.asarray(v).T)[0]
            if np.isnan(xxx):
                pass
            else:
                hd.append(xxx)
        try:
            accumulated_HD.append(np.max(hd))
        except:
            pass # if no matching segment found continue
    return np.mean(accumulated_HD)

def dice(gt_map, predicted_map):
    im1 = gt_map>0
    im2 = predicted_map>0
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())


