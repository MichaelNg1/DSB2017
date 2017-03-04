"""
Author: Neil Jassal
Email: neil.jassal@gmail.com

Updated 3/3/2017

Preprocessing of DSB 2017 data. Adapted from:
https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
"""
import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, csv I/O
import dicom
import scipy.ndimage
import matplotlib.pyplot as plt
from IPython import embed

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

INPUT_FOLDER = "data\\sample_images\\"
patients = os.listdir(INPUT_FOLDER)  # unsorted
patients.sort()


def load_scan(path):
    """
    Loads scans in the given folder path
    @param path Filepath for the scan
    @return slices
    """
    slices = [dicom.read_file(path + '\\' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Fill in SliceThickness parameter
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                                 slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation -
                                 slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    """
    Rescale pixel values to fall within Hounsfield Units (HU). Areas outside
    scanning bounds are assigned -2000, and are rescaled to 0 (air).
    Hounsfield Unit values based on:
    https://en.wikipedia.org/wiki/Hounsfield_scale#The_HU_of_common_substances

    Conversion to HU done by multiplying by the rescale slope and adding the
    intercept. Both values are contained within scan metadata.

    @param slices List of slices containing metadata
    @return pixels HU-rescaled np-array image of a single scan
    """
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (values sometimes already int16). Values always <32k so
    #   conversion should be possible
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0 to approximate HU of air.
    # Intercept is usualy -1024, scaling air to ~0
    image[image == -2000] = 0

    # Convert to Hounsfield Units (HU)
    num_slices = len(slices)
    for slice_num in range(num_slices):
        intercept = slices[slice_num].RescaleIntercept
        slope = slices[slice_num].RescaleSlope

        if slope is not 1:
            image[slice_num] = slope * image[slice_num].astype(np.float64)
            image[slice_num] = image[slice_num].astype(np.int16)

        image[slice_num] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    """
    Resample pixel spacing for different scans. Scans may have nonlinear pixel
    spacing, so the true distance between pixels can vary between scans.

    Equivalent spacing the dataset to a fixed isotroipc resolution allows for
    using 3D convnets or other learning techniques without also having to
    learn or account for zoom/slice invariance.

    WARNING: This function runs slowly

    @param image Numpy array containing image pixels
    @param scan Scan metadata object
    @param new_spacing Updated spacing for 3D scan

    @return image Image resampled to have updated pixel spacing
    @return new_spacing Updated pixel spacing TODO update description
    """
    # Get current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing,
                       dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor,
                                             mode='nearest')
    return image, new_spacing


def plot_3d(image, threshold=-300):
    """
    Generates and displays a 3D plot of the scan in an upright position.
    @param image The 3D image to plot
    @param threshold The threhold used to generate vertices and faces
    """
    # Position scan upright so patient head faces top of camera
    p = image.transpose(2, 1, 0)
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Indexing: 'verts[faces]' to generate collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def largest_label_volume(image, bg=-1):
    """
    Find the largest labeled type in the image (air)
    @param image The image to find the largest volume
    @param bg

    @return label with the largest volume in the scan
    """
    val, counts = np.unique(image, return_counts=True)

    counts = counts[vals != bg]
    vals= vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    """
    TODO add comment
    """
    pass


if __name__ == "__main__":
    first_patient = load_scan(INPUT_FOLDER + patients[0])
    first_patient_pixels = get_pixels_hu(first_patient)
    # plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
    # plt.xlabel("Hounsfield Units (HU)")
    # plt.ylabel("Frequency")
    # plt.show()

    # Show some slice in the middle
    # plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
    # plt.show()

    pix_resampled, spacing = resample(first_patient_pixels, first_patient,
                                      [1, 1, 1])
    print("Shape before resampling\t", first_patient_pixels.shape)
    print("Shape after resampling\t", pix_resampled.shape)

    # plot_3d(pix_resampled, 400)

