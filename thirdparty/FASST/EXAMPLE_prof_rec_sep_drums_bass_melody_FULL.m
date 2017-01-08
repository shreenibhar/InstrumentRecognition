function EXAMPLE_prof_rec_sep_drums_bass_melody_FULL(file_name, data_dir, results_dir)

%
% EXAMPLE_prof_rec_sep_drums_bass_melody_FULL(file_name, data_dir, results_dir);
%
% This scripts cuts the recording in small parts, and apply 
% EXAMPLE_prof_rec_sep_drums_bass_melody function to them
% (EXAMPLE_prof_rec_sep_drums_bass_melody cannot process a full track of
% due to memory limitations in Matlab)
%
% Example of using FASST for separation of professionally produced (stereo)
% music recordings. Thefollowing four stereo components are extracted from
% the mix:
%     - drums
%     - bass
%     - main melody
%     - the rest
%
%
% input 
% -----
%
% file_name         : file name
% data_dir          : input data directory
% results_dir       : output results directory
%
% output
% ------
%
% estimated source images are written in the results_dir
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                        
% Flexible Audio Source Separation Toolbox (FASST), Version 1.0                                    
%                                                                                                  
% Copyright 2011 Alexey Ozerov, Emmanuel Vincent and Frederic Bimbot                               
% (alexey.ozerov -at- inria.fr, emmanuel.vincent -at- inria.fr, frederic.bimbot -at- irisa.fr)     
%                                                                                                  
% This software is distributed under the terms of the GNU Public License                           
% version 3 (http://www.gnu.org/licenses/gpl.txt)                                                  
%                                                                                                  
% If you use this code please cite this research report                                            
%                                                                                                  
% A. Ozerov, E. Vincent and F. Bimbot                                                              
% "A General Flexible Framework for the Handling of Prior Information in Audio Source Separation," 
% IEEE Transactions on Audio, Speech and Signal Processing 20(4), pp. 1118-1133 (2012).                            
% Available: http://hal.inria.fr/hal-00626962/                                                     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                        


fs_def = 44100;
nchan_def = 2;
sep_segm_len = 20 * fs_def;        % length of separated segment (in samples)
last_segm_min_len = 10 * fs_def;   % minimum length of last segment (in samples)
overlap_len = 6000;                % length of overlap (in samples)

smoothing_win = ones(nchan_def, sep_segm_len + overlap_len);
for k = 1:overlap_len
    smoothing_win(:, k) = k / overlap_len;
    smoothing_win(:, sep_segm_len + k) = smoothing_win(:, sep_segm_len + k) - smoothing_win(:, k);
end;

% add '/' character at the end of directory names
if data_dir(end) ~= '/', data_dir = [data_dir, '/']; end;
if results_dir(end) ~= '/', results_dir = [results_dir, '/']; end;

input_file_name = [data_dir, file_name];
results_file_name = [results_dir, file_name];

a = wavread(input_file_name, 'size');
tot_sig_len = a(1);
nchan = a(2);
[x, fs, nbits] = wavread(input_file_name, [1 3]);
if fs ~= fs_def || nchan ~= 2
    error('Incorrect sampling frequency or number of channels.');
end;

last_seg_flag = 0;

segm_ind = 1;

seg_beg = 1;
seg_end = sep_segm_len + overlap_len;

seg_beg_arr = [];
seg_end_arr = [];

src1_part_fn_arr = [];
src2_part_fn_arr = [];
src3_part_fn_arr = [];
src4_part_fn_arr = [];

approx_segm_num = round(tot_sig_len / sep_segm_len);

while ~last_seg_flag
    fprintf('Processing segment %d of total %d\n', segm_ind, approx_segm_num);

    if seg_end + last_segm_min_len >= tot_sig_len
        seg_end = tot_sig_len;
        last_seg_flag = 1;
    end;

    seg_beg_arr = [seg_beg_arr, seg_beg];
    seg_end_arr = [seg_end_arr, seg_end];

    x = wavread(input_file_name, [seg_beg seg_end]);
    x = x.';
    mix_nsamp = size(x,2);

    % windowing
    if seg_beg == 1 && seg_end == tot_sig_len  % first and last segment
        % no windowing
    elseif seg_beg == 1                        % first segment
        x(:,overlap_len+1:end) = x(:,overlap_len+1:end) .* smoothing_win(:,overlap_len+1:end);
    elseif seg_end == tot_sig_len              % last segment
        x(:,1:overlap_len) = x(:,1:overlap_len) .* smoothing_win(:,1:overlap_len);
    else
        x = x .* smoothing_win;
    end;

    part_mix_name = [file_name(1:end-4), '__part_' num2str(segm_ind) '.wav'];

    src1_part_fn_arr{segm_ind} = [results_dir, part_mix_name(1:end-4), '__est_melody.wav'];
    src2_part_fn_arr{segm_ind} = [results_dir, part_mix_name(1:end-4), '__est_drums.wav'];
    src3_part_fn_arr{segm_ind} = [results_dir, part_mix_name(1:end-4), '__est_bass.wav'];
    src4_part_fn_arr{segm_ind} = [results_dir, part_mix_name(1:end-4), '__est_other.wav'];

    % SEPARATION
    wavwrite(x.', fs, nbits, [results_dir, part_mix_name]);
    EXAMPLE_prof_rec_sep_drums_bass_melody(part_mix_name, results_dir, results_dir);
    delete([results_dir, part_mix_name]);

    seg_beg = seg_beg + sep_segm_len;
    seg_end = seg_end + sep_segm_len;

    segm_ind = segm_ind + 1;
end;

src1_fn = [results_file_name(1:end-4), '__est_melody.wav'];
src2_fn = [results_file_name(1:end-4), '__est_drums.wav'];
src3_fn = [results_file_name(1:end-4), '__est_bass.wav'];
src4_fn = [results_file_name(1:end-4), '__est_other.wav'];

% reconstruct full track sources
x = zeros(tot_sig_len, nchan);
for segm_ind = 1:length(seg_beg_arr)
    part_file_name_src = src1_part_fn_arr{segm_ind};
    x(seg_beg_arr(segm_ind):seg_end_arr(segm_ind), :) = x(seg_beg_arr(segm_ind):seg_end_arr(segm_ind), :) + wavread(part_file_name_src);
end;
wavwrite(x, fs, nbits, src1_fn);
clear('x');

x = zeros(tot_sig_len, nchan);
for segm_ind = 1:length(seg_beg_arr)
    part_file_name_src = src2_part_fn_arr{segm_ind};
    x(seg_beg_arr(segm_ind):seg_end_arr(segm_ind), :) = x(seg_beg_arr(segm_ind):seg_end_arr(segm_ind), :) + wavread(part_file_name_src);
end;
wavwrite(x, fs, nbits, src2_fn);
clear('x');

x = zeros(tot_sig_len, nchan);
for segm_ind = 1:length(seg_beg_arr)
    part_file_name_src = src3_part_fn_arr{segm_ind};
    x(seg_beg_arr(segm_ind):seg_end_arr(segm_ind), :) = x(seg_beg_arr(segm_ind):seg_end_arr(segm_ind), :) + wavread(part_file_name_src);
end;
wavwrite(x, fs, nbits, src3_fn);
clear('x');

x = zeros(tot_sig_len, nchan);
for segm_ind = 1:length(seg_beg_arr)
    part_file_name_src = src4_part_fn_arr{segm_ind};
    x(seg_beg_arr(segm_ind):seg_end_arr(segm_ind), :) = x(seg_beg_arr(segm_ind):seg_end_arr(segm_ind), :) + wavread(part_file_name_src);
end;
wavwrite(x, fs, nbits, src4_fn);
clear('x');

% delete temporary files
for segm_ind = 1:length(seg_beg_arr)
    delete(src1_part_fn_arr{segm_ind});
    delete(src2_part_fn_arr{segm_ind});
    delete(src3_part_fn_arr{segm_ind});
    delete(src4_part_fn_arr{segm_ind});
end;
