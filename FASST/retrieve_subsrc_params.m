function [Vs, As, rank_part_ind] = retrieve_subsrc_params(mix_str)

%
% [Vs, As, rank_part_ind] = retrieve_subsrc_params(mix_str);
%
% retrieve sub-sources parameters (in matrix form)
%
%
% input
% -----
%
% mix_str           : input mix structure
% 
%
% output
% ------
%
% Vs                : [F, N, R] spectral power of sub-sources
% As                : [M, F, R] mixing coefficients of sub-source vectors
% rank_part_ind     : K-length array of spatial components indices rank partition
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


[F, N, M, M] = size(mix_str.Cx);

K = length(mix_str.spat_comps);

R = 0;

rank_part_ind = cell(1, K);

% compute sum pf ranks and rank partition
for j = 1:K
    spat_comp = mix_str.spat_comps{j};

    rank = size(spat_comp.params, 2);
    
    R = R + rank;
    
    if j == 1
        rank_part_ind{j} = (1:rank);
    else
        rank_part_ind{j} = rank_part_ind{j-1}(end) + (1:rank);
    end;
end;

Vs = zeros(F, N, R);
As = zeros(M, F, R);

% fill in matrices
for j = 1:length(mix_str.spat_comps)
    spat_comp = mix_str.spat_comps{j};

    for r = rank_part_ind{j}
        Vs(:,:,r) = comp_spat_comp_power(mix_str, j);
    end
    if strcmp(spat_comp.mix_type, 'inst')
        for f = 1:F
            As(:,f,rank_part_ind{j}) = spat_comp.params;
        end;
    else
        As(:,:,rank_part_ind{j}) = permute(spat_comp.params, [1 3 2]);
    end;
end;
