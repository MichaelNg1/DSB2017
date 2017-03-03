%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author:	Michael Nguyen
% Updated:	3/2/2017
% Email:	mn2769@columbia.edu
%
% view_samples visualizes the patient's CT scans in ascending order (height)
% Uses getkey.m by Jos van der Geest (jos@jasen.nl) - temporary
% (Caution: Keyboard commands are buggy)
%
% History
% Version 1.0: Displays only image
% Version 1.1: Computes a low rank representation of original image and error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

%% User defined Parameters %%
% Provide filepath to sample images and labels

% Michael's filepaths
% FILE_PATH = 'C:\Users\Michael\Documents\Kaggle\DSB2017\Sample';
% FILE_PATH_LABELS = 'C:\Users\Michael\Documents\Kaggle\DSB2017\stage1_labels.csv';

% Neil's filepaths
FILE_PATH = 'data\sample_images';
FILE_PATH_LABELS = 'data\stage1_labels.csv';

%% File and Processing Parameters
EX_PATIENT_NO_CANCER = '0a0c32c9e08cc2ea76a71649de56be6d';
EX_PATIENT_CANCER = '0d06d764d3c07572074d468b4cff954f';
PATIENT_NAME = EX_PATIENT_CANCER;

% Specify type of display
% 1 = standard slice, 2 = low rank representation with error
DISPLAY = 2;

VARIANCE = 0.9;

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
%     plot_low_rank(i, PATIENT_CANCER_LABEL, PATIENT_NAME, SAMPLE_FILES_LIST);

    fig = figure(1);
    [X, map] = dicomread(SAMPLE_FILES_LIST{i});
    if DISPLAY == 1 % display individual slice
        imshow(X,map)        
    elseif DISPLAY == 2
        subplot(1,3,1);
        imshow(X, map);
        % Evaluate low rank representation
        [U,S,V] = svd(double(X));
        sing_vals = diag(S);
        sing_vals_sum = sum(sing_vals);
        temp = 1;
        while(sum(sing_vals(1:temp)) < VARIANCE * sing_vals_sum)
            temp = temp + 1;
        end
        subplot(1,3,2);
        sparse_X = U(:,1:temp) * S(1:temp, 1:temp)* V(:,1:temp)';
        imshow(sparse_X, map);
        title(['Singular Values Kept: ' num2str(temp) '/' num2str(length(sing_vals))]);
        xlabel(['Percentage Variance Maintained: ' num2str(VARIANCE)])

        % Visualize the error (optional: choose which colormap to use)
        subplot(1,3,3)
        error = num2str(norm(sparse_X - double(X),'fro'));
        imshow(abs(uint16(sparse_X) - X), jet);
        imshow(abs(uint16(sparse_X) - X), map);
        colorbar;
        title('Absolute Error');
        xlabel(['l2 error: ' error])
    end
    
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
