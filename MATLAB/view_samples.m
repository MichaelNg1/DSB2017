%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author:	Michael Nguyen
% Updated:	3/2/2017
% Email:	mn2769@columbia.edu
%
% view_samples visualizes the patient's CT scans in ascending order (height)
% Uses getkey.m by Jos van der Geest (jos@jasen.nl) - temporary
% (Caution: Keyboard commands are buggy)
%
% Version 1.0 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

%% User defined Parameters %%
% Provide filepath to sample images and labels
FILE_PATH = 'C:\Users\Michael\Documents\Kaggle\DSB2017\Sample';
FILE_PATH_LABELS = 'C:\Users\Michael\Documents\Kaggle\DSB2017\stage1_labels.csv';
PATIENT_NAME = '0d06d764d3c07572074d468b4cff954f';

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

%% View the images in ascending order %%
fprintf(['Press a or d to decrement/increment, respectively.' ...
	'\nPress s to enter a custom index. \nPress esc to exit. \n']);

i = 1;
while(true)

	% Plot figure
	[X, map] = dicomread(SAMPLE_FILES_LIST{i});

	fig = figure(1); 
	imshow(X,map);

	title(['Sample ' num2str(i) ' out of ' num2str(length(SAMPLE_FILES_LIST))]);
	xlabel(['Patient: ' PATIENT_NAME ' | Cancer: ' PATIENT_CANCER_LABEL{1}]);
	ylabel(['z: ' num2str(img_pos(i,3))]);
	drawnow;

	% Keyboard commands
	ch = getkey('non-ascii');

	if (strcmp(ch,'escape'))
		return;
	elseif (strcmp(ch,'a'))
		i = i - 1;
	elseif (strcmp(ch,'d'))
		i = i + 1;
	elseif (strcmp(ch,'s'))
		i = input('\nEnter index: ');
	end

	% Ensure we stay in range
	i = max(1, i);
	i = min(i, length(SAMPLE_FILES_LIST));
end
