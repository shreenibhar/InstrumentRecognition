function ie = separate_spec_comps(x, mix_str, sep_cmp_inds)

%
% ie = separate_spec_comps(x, mix_str, sep_cmp_inds);
%
% separate spectral components
%
%
% input
% -----
%
% x                 : [nchan x nsampl] mixture signal
% mix_str           : input mix structure
% sep_cmp_inds      : (opt) array of indices for components to separate
%                     (def = {1, 2, ..., K_spec})
% 
%
% output
% ------
%
% ie                : [K_sep x nsampl x nchan] estimated spectral components images,
%                     where K_sep = length(sep_cmp_inds) is the number of
%                     components to separate
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


K_spec = length(mix_str.spec_comps);

if nargin < 3 || isempty(sep_cmp_inds)
    sep_cmp_inds = cell(1, K_spec);
    for j = 1:K_spec
        sep_cmp_inds{j} = j;
    end;
end;

[nchan, nsampl] = size(x);

K_sep = length(sep_cmp_inds);

WG = comp_WG_spec_comps(mix_str, sep_cmp_inds);

ie = zeros(K_sep, nsampl, nchan);

for j = 1:K_sep
    if strcmp(mix_str.transf, 'stft')
        ie(j,:,:) = filter_stft(x.',WG(:,:,:,:,j));
    else % 'qerb'
        ie(j,:,:) = filter_qerbt(x.',WG(:,:,:,:,j),mix_str.fs,mix_str.wlen);
    end;
end;
