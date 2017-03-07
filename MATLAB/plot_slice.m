%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author:	 Neil Jassal
% Updated:	3/3/2017
% Email:	neil.jassal@gmail.com
%
% plot_slice Generates a figure for the given sample slice. Depending on
% input parameters specified in view_samples.m, it will display either the
% raw image or a low-rank representation. The low rank representation
% includes the raw image, low-rank image, and a map of the error between
% the two.
%
% @param index The slice index to plot
%
% History
% Version 1.0: Plots raw image or low-rank representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ ] = plot_slice(index)
%PLOT_SLICE Creates a plot of the specified slice of the DICOM file.
global DISPLAY PATIENT_INFO SAMPLE_FILES_LIST VARIANCE;

[X, map] = dicomread(SAMPLE_FILES_LIST{index});
fig_name = ['Sample ' num2str(index) ' out of ' num2str(length(SAMPLE_FILES_LIST))];

%% Display individual slice
if DISPLAY == 1
    imshow(X,map)
    title(fig_name);
    xlabel(patient_info);
    ylabel(['z: ' num2str(img_pos(index,3))]);

%% Display Low-rank representation
elseif DISPLAY == 2 % Display low rank representation
    % Evaluate low rank representation sparse_X
    [U,S,V] = svd(double(X));
    sing_vals = diag(S);
    sing_vals_sum = sum(sing_vals);
    temp = 1;
    while(sum(sing_vals(1:temp)) < VARIANCE * sing_vals_sum)
        temp = temp + 1;
    end
    sparse_X = U(:,1:temp) * S(1:temp, 1:temp)* V(:,1:temp)';

    % Compute the error (optional: choose which colormap to use)
    error = num2str(norm(sparse_X - double(X),'fro'));

    % Visualize original, low rank representation, and error
    % set(gcf, 'Position', [600 600 800 400]);
  
    subplot(1,3,1); % original
    imshow(X, map);
    title('Original Slice');

    subplot(1,3,2); % low rank representation
    imshow(sparse_X, map);
    title(['Singular Values Kept: ' num2str(temp) '/' num2str(length(sing_vals))]);
    xlabel(['Percentage Variance Maintained: ' num2str(VARIANCE)])      

    subplot(1,3,3) % error
    imshow(abs(uint16(sparse_X) - X), jet);
    imshow(abs(uint16(sparse_X) - X), map);
    colorbar;
    title('Absolute Error');
    xlabel(['L2 error: ' error])        
    suplabel([fig_name, PATIENT_INFO],'t', [.075 .075 .85 .85])
end
drawnow;

end

