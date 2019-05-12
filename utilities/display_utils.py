# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:47:50 2019

Plotting and display utilities for 2019 POGO Workshop on Machine Learning in Biological Oceanography

@author: eric (e1orenst@ucsd.edu)
"""

import sys
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import cv2


def make_confmat(labels, decs, acc, mat_size=8, outpath=None):
    """
    takes classifier output and labels to generate a confusion matrix
    :param labels: list of true numeric labels
    :param dec: list of labels from classifiers
    :param acc: accuracy [float]
    :param mat_size: size for plot in inches (assumed square dims) [int]
    :param outpath: file path for saving [str]
    """
    # compute the confusion matrix from sklearn
    conf = confusion_matrix(labels, decs)

    # go through and normalize the output
    norm_conf = []
    for ii in conf:
        aa = 0
        tmp_arr = []
        aa = sum(ii, 0)
        if aa == 0:
            for jj in ii:
                tmp_arr.append(0)
        else:
            for jj in ii:
                tmp_arr.append(float(jj)/float(aa))
        norm_conf.append(tmp_arr)

    # create the figure output
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')
    plt.title('Object accuracy: %.3f' % acc)
    cb = fig.colorbar(res)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.set_size_inches(mat_size, mat_size)

    # turn off this block of comments to remove numeric ticks
    """
    plt.tick_params(
    axis = 'both',
    which = 'both',
    bottom = 'off',
    top = 'off',
    labelbottom = 'off',
    right = 'off',
    left = 'off',
    labelleft = 'off')
    """

    # if a file for saving is defined, save it
    if outpath:
        plt.savefig(outpath, dpi=120)

    # otherwise, display it
    else:
        plt.show()


def imshow_tensor(tensor, *args, **kwargs):
    """
    Display a PyTorch tensor using matplotlib.
    """
    plt.imshow(tensor.permute(1, 2, 0).numpy(), *args, **kwargs)

def aspect_resize(im, ii=226):

    """
    image == input array
    ii == desired dimensions
    """

    cen = np.floor(np.array((ii, ii))/2.0).astype('int')
    dim = im.shape[0:2]

    if dim[0] != dim[1]:
        # get the largest dimension
        large_dim = max(dim)
        
        # ratio between the large dimension and required dimension
        rat = float(ii)/large_dim
        
        # get the smaller dimension that maintains the aspect ratio
        small_dim = int(min(dim)*rat)
        
        # get the indicies of the large and small dimensions
        large_ind = dim.index(max(dim))
        small_ind = dim.index(min(dim))
        dim = list(dim)
        
        # the dimension assigment may seem weird cause of how python indexes images
        dim[small_ind] = ii
        dim[large_ind] = small_dim
        dim = tuple(dim)

        im = cv2.resize(im, dim)
        half = np.floor(np.array(im.shape[0:2])/2.0).astype('int')
        
        # make an empty array, and place the new image in the middle
        res = np.zeros((ii, ii, 3), dtype='uint8')
        
        if large_ind == 1:
            test = res[cen[0]-half[0]:cen[0]+half[0], cen[1]-half[1]:cen[1]+half[1]+1]
            if test.shape != im.shape:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]+1] = im
            else:
                res[cen[0]-half[0]:cen[0]+half[0], cen[1]-half[1]:cen[1]+half[1]+1] = im
        else:
            test = res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]]
            if test.shape != im.shape:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]+1] = im
            else:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]] = im
    else:
        res = cv2.resize(im, (ii, ii))

    return res


def tile_images(images, tile_dim, resize=128):
    """
    takes a list of images and tiles them
    :param images: input list of image paths
    :param tile_dim: number to tile in each dimension [hh x ww] as int
    :param resize: size to resize the input images
    :return:
    """

    out = np.zeros((resize*tile_dim[0], resize*tile_dim[1], 3))
    out = out.astype(np.uint8)

    for idx, img in enumerate(images):
        ii = idx % tile_dim[1]
        jj = idx // tile_dim[1]

        im_in = cv2.imread(img)
        im_in = cv2.cvtColor(im_in, cv2.COLOR_BGR2RGB)
        im_out = aspect_resize(im_in, resize)

        out[jj*resize:jj*resize+resize, ii*resize:ii*resize+resize, :] = im_out

    return out
