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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_confmat(labels, decs, acc, outpath=None):
    """
    takes classifier output and labels to generate a confusion matrix
    :param labels: list of true numeric labels
    :param dec: list of labels from classifiers
    :param acc: accuracy [float]
    :param outpath: file path for saving [str]
    """
    # compute the confusion matrix from sklearn
    conf = confusion_matrix(labels,decs)

    # go through and normalize the output
    norm_conf = []
    for ii in conf:
        aa = 0
        tmp_arr = []
        aa = sum(ii,0)
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
    res = ax.imshow(np.array(norm_conf), cmap = plt.cm.jet, interpolation='nearest')
    plt.title('Object accuracy: %.3f' % acc)
    cb = fig.colorbar(res)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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
