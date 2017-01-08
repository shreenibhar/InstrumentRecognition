function ie = separate_spat_comps(x, mix_str)

%
% ie = separate_spat_comps(x, mix_str);
%
% separate spatial components
%
%
% input
% -----
%
% x                 : [nchan x nsampl] mixture signal
% mix_str           : input mix structure
% 
%
% output
% ------
%
% ie                : [nsrc x nsampl x nchan] estimated source images
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

[nchan, nsampl] = size(x);

nsrc = length(mix_str.spat_comps);

WG = comp_WG_spat_comps(mix_str);

ie = zeros(nsrc, nsampl, nchan);

for j = 1:nsrc
    if strcmp(mix_str.transf, 'stft')
        ie(j,:,:) = filter_stft(x.',WG(:,:,:,:,j));
    else % 'qerb'
        ie(j,:,:) = filter_qerbt(x.',WG(:,:,:,:,j),mix_str.fs,mix_str.wlen);
    end;
end;
