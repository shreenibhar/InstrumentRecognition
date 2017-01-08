function ie = separate_NMF_comp_group(x, mix_str, NMF_ind_arr, factor_ind)

%
% ie = separate_NMF_comp_group(x, mix_str, NMF_ind_arr, factor_ind);
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

% separate a group of NMF components
%
%
% input
% -----
%
% x                 : [nchan x nsampl] mixture signal
% mix_str           : input mix structure
% NMF_ind_arr       : cell array of separated NMF components flags (1 = to extract)
% factor_ind         : factor index
% 
%
% output
% ------
%
% ie                : [nsampl x nchan] estimated spectral component image
%


WG = comp_WG_NMF_comp_group(mix_str, NMF_ind_arr, factor_ind);

if strcmp(mix_str.transf, 'stft')
    ie = filter_stft(x.',WG);
else % 'qerb'
    ie = filter_qerbt(x.',WG,mix_str.fs,mix_str.wlen);
end;
