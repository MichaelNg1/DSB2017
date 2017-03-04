"""
Author: Neil Jassal
Email: neil.jassal@gmail.com

Updated 3/3/2017

Preprocessing of DSB 2017 data. Based on:
https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
"""
import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, csv I/O
import dicom
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # TODO need this?

INPUT_FOLDER = "data\\sample_images\\"
patients = os.listdir(INPUT_FOLDER).sort()
