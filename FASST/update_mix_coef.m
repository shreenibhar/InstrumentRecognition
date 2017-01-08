function [mix_str, As_upd] = update_mix_coef(mix_str_inp, hat_Rxs, hat_Rss, As, rank_part_ind)

%
% [mix_str, As_upd] = update_mix_coef(mix_str_inp, hat_Rxs, hat_Rss, As, rank_part_ind);
%
% update rank-1 mixing coefficients
%
%
% input
% -----
%
% mix_str_inp       : input mixture structure
% hat_Rxs           : [F,2,R] expected Rxs suffitient statistics
% hat_Rss           : [F,R,R] expected Rss suffitient statistics
% As                : [M, F, R] mixing coefficients of sub-source vectors
% rank_part_ind     : K-length array of spatial components indices rank partition
% 
%
% output
% ------
%
% mix_str           : updated output mixture structure
% As_upd            : updated [M, F, K1] mixing coefficients of rank-1 sources
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


mix_str = mix_str_inp;

[M, F, R] = size(As);

K = length(mix_str.spat_comps);

upd_inst_ind = [];
upd_inst_other_ind = [];
upd_conv_ind = [];
upd_conv_other_ind = [];

for k = 1:K
    spat_comp = mix_str.spat_comps{k};
    
    if strcmp(spat_comp.mix_type, 'inst') && strcmp(spat_comp.frdm_prior, 'free')
        upd_inst_ind = [upd_inst_ind, rank_part_ind{k}];
    else
        upd_inst_other_ind = [upd_inst_other_ind, rank_part_ind{k}];
    end;
    
    if strcmp(spat_comp.mix_type, 'conv') && strcmp(spat_comp.frdm_prior, 'free')
        upd_conv_ind = [upd_conv_ind, rank_part_ind{k}];
    else
        upd_conv_other_ind = [upd_conv_other_ind, rank_part_ind{k}];
    end;
end;


As_upd = As;


% update linear instantenous coefficients
K_inst = length(upd_inst_ind);

hat_Rxs_bis = zeros(F, 2, K_inst);

for f = 1:F
    hat_Rxs_bis(f,:,:) = shiftdim(hat_Rxs(f,:,upd_inst_ind), 1) - ...
        reshape(As_upd(:,f,upd_inst_other_ind), size(As_upd, 1), length(upd_inst_other_ind)) * ...
        shiftdim(hat_Rss(f,upd_inst_other_ind, upd_inst_ind), 1);
end;
rm_hat_Rxs_bis = real(squeeze(mean(hat_Rxs_bis, 1)));
rm_hat_Rss = real(squeeze(mean(hat_Rss(:,upd_inst_ind, upd_inst_ind), 1)));

As_upd_inst = zeros(M, K_inst);
As_upd_inst(:) = rm_hat_Rxs_bis * inv(rm_hat_Rss);  % NOTE: there is a matrix inversion

for f = 1:F
    As_upd(:,f,upd_inst_ind) = As_upd_inst;
end;


% update convolutive coefficients
K_conv = length(upd_conv_ind);

hat_Rxs_bis = zeros(F, 2, K_conv);

for f = 1:F
    hat_Rxs_bis(f,:,:) = shiftdim(hat_Rxs(f,:,upd_conv_ind), 1) - ...
        reshape(As_upd(:,f,upd_conv_other_ind), size(As_upd, 1), length(upd_conv_other_ind)) * ...
        shiftdim(hat_Rss(f, upd_conv_other_ind, upd_conv_ind), 1);
end;

for f = 1:F
    As_upd(:,f,upd_conv_ind) = squeeze(hat_Rxs_bis(f,:,:)) * inv(squeeze(hat_Rss(f,upd_conv_ind, upd_conv_ind)));  % NOTE: there is a matrix inversion
end;

As_upd = permute(As_upd, [1 3 2]);

% update parameters in the structure
for k = 1:K
    spat_comp = mix_str.spat_comps{k};
    
    if strcmp(spat_comp.frdm_prior, 'free')
        if strcmp(spat_comp.mix_type, 'inst')
            spat_comp.params = mean(As_upd(:,rank_part_ind{k}, :), 3);
        else
            spat_comp.params = As_upd(:,rank_part_ind{k},:);
        end;
    end;
    
    mix_str.spat_comps{k} = spat_comp;
end;
