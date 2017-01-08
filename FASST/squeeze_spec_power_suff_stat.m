function hat_W = squeeze_spec_power_suff_stat(hat_Ws, rank_part_ind)

%
% hat_W = squeeze_spec_power_suff_stat(hat_Ws, rank_part_ind);
%
% update full rank spatial covariance matrices
%
%
% input
% -----
%
% hat_Ws            : [F,N,R] expected S-spectral (subsources) power suffitient statistics 
% rank_part_ind     : K-length array of spatial components indices rank partition
% 
%
% output
% ------
%
% hat_W             : [F,N,K] expected (joint) spectral power suffitient statistics
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


[F, N, R] = size(hat_Ws);

K = length(rank_part_ind);

hat_W = zeros(F,N,K);

for k = 1:K
    hat_W(:,:,k) = mean(hat_Ws(:,:,rank_part_ind{k}), 3);
end;
