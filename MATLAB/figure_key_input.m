%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author:	 Neil Jassal
% Updated:	3/3/2017
% Email:	neil.jassal@gmail.com
%
% figure_key_input is a callback function enabled by keyboard input while a
% figure is selected. Allows for moving between samples, selecting sample
% indices to display, and exiting program function.
%
% Controls:
% a - display previous sample
% d - display next sample
% s - select sample within sample range
% esc - terminate program

% @param fig Figure object that called the function
% @param event Callback data object containing character information
% @param index The slice index to plot
%
% History
% Version 1.0: Key input for moving between samples, replaces getkey()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ ] = figure_key_input(fig, event, index)
%FIGURE_KEY_INPUT Callback to parse keyboard input from figure.
global SAMPLE_FILES_LIST index;

ch = event.Character;

if (strcmp(ch,'a'))
    index = index - 1;
elseif (strcmp(ch,'d'))
    index = index + 1;
elseif (strcmp(ch,'s'))
    index = input(['\nEnter index (1-' num2str(length(SAMPLE_FILES_LIST)) '): ']);
elseif(strcmp(get(fig, 'CurrentKey'), 'escape'))
    close(1);
    return;
end

% Ensure i is in range
index = max(1, index);
index = min(index, length(SAMPLE_FILES_LIST));

plot_slice(index);

end

