function [hat_W, log_like] = comp_suff_stat_M1(mix_str)

%
% [hat_W, log_like] = comp_suff_stat_M1(mix_str);
%
% compute suffitient statistics for GEM algorithm in the 1 channel case
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
% hat_W             : [F, N, K_tot] expected S-spectral power suffitient statistics
% log_like          : log-likelihood
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


if M ~= 1
    error('comp_suff_stat_M1 function is only implemented for M = 1');
end;


CLMN = ones(1,N);

Sigma_x = mix_str.Noise_PSD * CLMN;

K_tot = length(mix_str.spat_comps);

Sigma_comps = zeros(K_tot,F,N);

R_full = zeros(K_tot,F,N);

for k = 1:K_tot
    spat_comp = mix_str.spat_comps{k};

    V = comp_spat_comp_power(mix_str, k);

    A = spat_comp.params;
    if strcmp(spat_comp.mix_type, 'inst')
        A = repmat(A, [1, 1, F]);
    end;

    rank = size(A, 2);

    for r = 1:rank
        R_full_loc = R_full(k,:,:);
        R_full_loc(1,:,:) = abs(squeeze(A(1,r,:))).^2 * CLMN;
        
        R_full(k,:,:) = R_full(k,:,:) + R_full_loc;
    end;

    Sigma_comps(k,:,:) = squeeze(R_full(k,:,:)) .* V;

    Sigma_x = Sigma_x + squeeze(Sigma_comps(k,:,:));
end;


% compute log-likelihood
log_like = -sum(sum(log(Sigma_x * pi) + (mix_str.Cx(:,:,1,1) ./ Sigma_x) )) / (N * F);


hat_W = zeros(F, N, K_tot);

for k_tot = 1:K_tot
    Sigma_comps_loc = squeeze(Sigma_comps(k_tot,:,:));
    
    % compute Wiener gain
    G_sac = Sigma_comps_loc ./ Sigma_x;

    hat_W(:, :, k_tot) = real((abs(G_sac).^2 .* mix_str.Cx(:,:,1,1) + Sigma_comps_loc - G_sac .* Sigma_comps_loc) ./ squeeze(R_full(k_tot,:,:)));
end;
