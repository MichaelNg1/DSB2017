%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author:	Michael Nguyen, Neil Jassal
% Updated:	3/3/2017
% Email:	mn2769@columbia.edu, neil.jassal@gmail.com
%
% view_samples visualizes the patient's CT scans in ascending order (height)
%
% History
% Version 1.0: Displays only image
% Version 1.1: Computes a low rank representation of original image and error
% Version 1.1: Options for image or low-rank representation, replaced
%               getkey() with callbacks for cleaner UI navigation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
global DISPLAY PATIENT_INFO SAMPLE_FILES_LIST VARIANCE index;

%% User defined Parameters %%
% Provide filepath to sample images and labels

% Michael's filepaths
% FILE_PATH = 'C:\Users\Michael\Documents\Kaggle\DSB2017\Sample';
% FILE_PATH_LABELS = 'C:\Users\Michael\Documents\Kaggle\DSB2017\stage1_labels.csv';

% Neil's filepaths
FILE_PATH = 'data\sample_images';
FILE_PATH_LABELS = 'data\stage1_labels.csv';

% Patient and Processing Parameters
EX_PATIENT_NO_CANCER = '0a0c32c9e08cc2ea76a71649de56be6d';
EX_PATIENT_CANCER = '0d06d764d3c07572074d468b4cff954f';
PATIENT_NAME = EX_PATIENT_CANCER;

VARIANCE = 0.9; % Only for low rank representation

% Specify type of display
% 1 = standard slice, 2 = low rank representation with error
DISPLAY = 2;

%% The following lines is if the user wishes to define a custom index (3: NUM_SAMPLE)
% index = 4;
% SAMPLE_FOLDERS = dir(FILE_PATH);
% NUM_SAMPLE = length(SAMPLE_FOLDERS);
% PATIENT_NAME = SAMPLE_FOLDERS(index).name;

%% Set up path to the specific sample folder %%
SAMPLE_PATH = [FILE_PATH '\' PATIENT_NAME];
SAMPLE_FILES = dir(SAMPLE_PATH);
NUM_SAMPLE_FILES = length(SAMPLE_FILES);

%% Sort the images based on vertical height %%
% sample_info contains the metadata of the DICOM image
%	- sample_info.ImagePositionPatient contains the (x,y,z) orientation
img_pos = zeros(NUM_SAMPLE_FILES - 2, 3);
SAMPLE_FILES_LIST = cell(1, NUM_SAMPLE_FILES - 2);
for i = 3: NUM_SAMPLE_FILES
	SAMPLE_FILES_LIST{i-2} = [SAMPLE_PATH '\' SAMPLE_FILES(i).name];
	sample_info = dicominfo(SAMPLE_FILES_LIST{i-2});
	img_pos(i - 2, :) = sample_info.ImagePositionPatient;
end

[~, sorted_index] = sort(img_pos(:,3), 1);
img_pos = img_pos(sorted_index,:);
SAMPLE_FILES_LIST = SAMPLE_FILES_LIST(sorted_index);

%% Get the patient labels (0: no cancer, 1: cancer)
FileID = fopen(FILE_PATH_LABELS);
labels = textscan(FileID,'%s %s', 'Delimiter',',');
%labels = textscan(FileID,'%s %u8', 'Delimiter',',');
fclose(FileID);

temp_index = find( strcmp(labels{1}, PATIENT_NAME));
PATIENT_CANCER_LABEL = labels{2}(temp_index);
PATIENT_INFO = ['Patient: ' PATIENT_NAME ' | Cancer: ' PATIENT_CANCER_LABEL{1}];

%% Display figure and wait for callbacks
fprintf(['Press a or d to decrement/increment, respectively.' ...
	'\nPress s to enter a custom index. \nPress esc to exit. \n']);

index = 1;
display_fig = figure(1);
set(display_fig, 'KeyReleaseFcn', @figure_key_input);
plot_slice(index);

