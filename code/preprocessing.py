"""
Author: Neil Jassal
Email: neil.jassal@gmail.com

Updated 3/6/2017

Preprocessing of DSB 2017 data. Adapted from:
https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
"""
import os

import numpy as np  # linear algebra
import dicom
import scipy.ndimage

from IPython import embed
from skimage import measure, morphology

import visualize

class Preprocessor(object):
    """
    Preprocessing class for segmenting out the lungs.
    Processed images are saved as:
        [image, spacing, [cancer]]
    Train data has the cancer label as a boolean 0 or 1
    Test data has no label
    """

    def __init__(self, input_folder, labels_path=None):
        """
        Initialize by loading and sorting input folder, and csv file
        with labels.

        CSV file expects a header, and [name,cancer_id] pairs

        @param data_path Path to folder containing scans
        @param labels_path Filepath for csv containing labels.
        """
        self.input_folder = input_folder
        self.patients_list = os.listdir(input_folder)
        self.patients_list.sort()

        # Read csv labels
        self.labels = None
        if labels_path is not None:
            self.labels = dict()

            f = open(labels_path)
            for i, row in enumerate(f):
                if i is 0:  # Skip header row
                    continue
                row = row.strip().split(',')
                self.labels[row[0]] = int(row[1])

    def preprocess_all_scans(self, save, out_dir=None):
        """
        Runs preprocessing on all scans in the input folder, saves based
        on arguments. Wrapper around preprocess_single_scan()

        Saves each scan as out_dir/scan_name.npy
        The format saved is a numpy array of the following format:
            [image_data, scan_spacing, [cancer_id]]
        The cancer_id label is optional, and only exists for training data

        @param save Whether to save the preprocessed scans.
        @param out_dir Only required if saving, directory to output data
        """
        for scan_name in self.patients_list:
            self.preprocess_single_scan(scan_name, save, out_dir)

    def preprocess_single_scan(self, scan_name, save, out_dir=None):
        """
        Runs the preprocessing - includes loading data, rescaling, and
        segmenting lungs. If specified, saves output.

        @param scan_name String containing the name of the scan
        @param save Boolean - whether to save the output or not
        @param out_dir Only required if saving, directory to output data

        @return Boolean success status
        """
        # Create directory if it does not exist
        if save and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # If no cancer_id label is found, scan is test data
        cancer_id = None
        try:
            cancer_id = self.labels[scan_name]
        except:
            pass

        if save and out_dir is None:
            print("No output directory, unable to preprocess scan" + scan_name)
            return False

        # Run preprocessing pipeline
        scan = self._load_scan(self.input_folder + scan_name)
        scan_pixels = self._get_pixels_hu(scan)
        scan_resampled, spacing = self._resample(scan_pixels, scan, [1, 1, 1])

        scan_dilated = morphology.dilation(scan_resampled,
                                           morphology.ball(1))
        segmented_lungs = self._segment_lung_mask(scan_dilated, False)

        visualize.plot_3d(segmented_lungs, 0, save_dir=out_dir + scan_name)
        if save:
            save_array = [segmented_lungs, spacing]
            if cancer_id is not None:  # Only append id for training data
                save_array.append(cancer_id)
            np.save(out_dir + scan_name, save_array)

        return True

    def _load_scan(self, path):
        """
        Loads a single scan in the given folder path
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

    def _get_pixels_hu(self, slices):
        """
        Rescale pixel values to fall within Hounsfield Units (HU). Areas
        outside scanning bounds are assigned -2000, and are rescaled to 0 (air)
        Hounsfield Unit values based on:
        https://en.wikipedia.org/wiki/Hounsfield_scale#The_HU_of_common_substances

        Conversion to HU done by multiplying by the rescale slope and adding
        the intercept. Both values are contained within scan metadata.

        @param slices List of slices containing metadata
        @return pixels HU-rescaled np-array image of a single scan
        """
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (values sometimes already int16).
        # Values always <32k so conversion should be possible
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

    def _resample(self, image, scan, new_spacing=[1, 1, 1]):
        """
        Resample pixel spacing for different scans. Scans may have nonlinear
        pixel spacing, so the true distance between pixels can vary between
        scans.

        Equivalent spacing the dataset to a fixed isotroipc resolution allows
        for using 3D convnets or other learning techniques without also having
        to learn or account for zoom/slice invariance.

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

    def _largest_label_volume(self, image, bg=-1):
        """
        Find the largest labeled type in the image (air)
        @param image The image to find the largest volume
        @param bg Background value to find volume of

        @return label with the largest volume in the scan
        """
        vals, counts = np.unique(image, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    def _segment_lung_mask(self, image, fill_lung_structures=True,
                           threshold=-320):
        """
        Segments the lungs and area around it. Steps:
            1. Threshold image (empirically, -320 HU decent)
            2. connected components, fill air around person with 1s in binary
            3. Optional: for each slice, find largest connected component -
                body and air, set other values to 0. Fills in lungs in the mask
            4. Keep largest air pocket only

        @param image The image to segment the lungs from
        @param threshold Value to threshold the initial image by
        @param fill_lung_structures Whether to fill in structures in the lung
            when segmenting
        """
        # not binary, but 1 and 2. 0 = unwanted background
        binary_image = np.array(image > threshold, dtype=np.int8) + 1
        labels = measure.label(binary_image)

        # Pick corner pixel to detect which label corresponds to air
        #   Improvement: pick multiple background labels from around patient
        #   This is more resistant to 'trays' where patient lays, cutting air
        #   around patient in half
        background_label = labels[0, 0, 0]

        # Fill air around person
        binary_image[background_label == labels] = 2

        # Fill lung structures - better than morphological closing
        if fill_lung_structures:
            # For each slice, find largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = self._largest_label_volume(labeling, bg=0)

                if l_max is not None:  # Slice contains lung
                    binary_image[i][labeling != l_max] = 1

        binary_image -= 1  # Convert image back to 0/1 binary (from 1/2)
        binary_image = 1 - binary_image  # Invert so lungs are 1

        # Remove air pockets inside body
        labels = measure.label(binary_image, background=0)
        l_max = self._largest_label_volume(labels, bg=0)
        if l_max is not None:  # Air pockets exist
            binary_image[labels != l_max] = 0

        return binary_image

    def _normalize(self, image, min_bound=-1000.0, max_bound=400.0):
        """
        Normalizes the data between 0 and 1. Also applies given bounds
        as a minimum and maximum cutoff for the data.
        Default bounds are chosen from the common thresholds from LUNA16

        @param image The image to normalize
        @param min_bound Minimum normalization value
        @param max_bound Maximum normalization value
        @return Image normalized between min_bound and max_bound
        """
        image = (image - min_bound) / (max_bound - min_bound)
        image[image > 1] = 1.0
        image[image < 0] = 0.0
        return image

    def _zero_center(self, image, pixel_mean=0.25):
        """
        Zero centers the input image such that the mean value is 0.
        The default pixel mean is approximated using data from the LUNA16
        competition.
        A true value would average all values in the dataset and use the result
        as the pixel_mean.

        @param image The image to zero center
        @param pixel_mean The mean used to zero center the data.
        @return Zero centered image
        """
        image = image - pixel_mean
        return image


if __name__ == "__main__":
    LABELS_PATH = "data\\stage1_labels.csv"
    SAMPLE_INPUT_FOLDER = "data\\sample_images\\"
    SAMPLE_OUTPUT_FOLDER = "data\\sample_segmented_lungs\\"

    p = Preprocessor(SAMPLE_INPUT_FOLDER, LABELS_PATH)
    p.preprocess_all_scans(False, SAMPLE_OUTPUT_FOLDER)

    # p.preprocess_single_scan(p.patients_list[0], save=True,
    #                          out_dir=SAMPLE_OUTPUT_FOLDER)
