function EXAMPLE_ssep_Mult_NMF_inst()

data_dir    = 'example_data/';
result_dir  = 'example_data/';
file_prefix = 'Shannon_Hurley__Sunrise__inst_';

transf    = 'stft';
wlen      = 1024;
nsrc      = 3;     % number of sources
NMF_ncomp = 4;     % number of NMF components
iter_num  = 200;

% load mixture
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
fprintf('Input time-frequency representation\n');
[x, fs, nbins]=wavread([data_dir file_prefix '_mix.wav']);
x = x.';
mix_nsamp = size(x,2);

% compute time-frequency representation
Cx = comp_transf_Cx(x, transf, wlen, fs);

% fill in mixture structure
mix_str = init_mix_struct_Mult_NMF_inst(Cx, nsrc, NMF_ncomp, transf, fs, wlen);

% reinitialize mixing parameters
A = [sin(pi/8), sin(pi/4), sin(3*pi/8); cos(pi/8), cos(pi/4), cos(3*pi/8)];
for j = 1:nsrc
    mix_str.spat_comps{j}.params = A(:,j);
end;

% run parameters estimation (with simulated annealing)
mix_str = estim_param_a_post_model(mix_str, iter_num, 'ann');

% source separation
ie_EM = separate_spat_comps(x, mix_str);

% Computation of the spatial source images
fprintf('Computation of the spatial source images\n');
for j=1:nsrc,
    wavwrite(reshape(ie_EM(j,:,:),mix_nsamp,2),fs,nbins, ...
        [result_dir file_prefix '_sim_' int2str(j) '.wav']);
end
