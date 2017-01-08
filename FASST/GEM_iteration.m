function [mix_str, log_like] = GEM_iteration(mix_str_inp, upd_noise_flag)

%
% [mix_str, log_like] = GEM_iteration(mix_str_inp, upd_noise_flag);
%
% one itaration of GEM algorithm for
% a posteriori mixture model parameters estimation
%
%
% input
% -----
%
% mix_str_inp       : input mixture structure
% upd_noise_flag    : (opt) update noise PSD flag (def = 0)
% 
%
% output
% ------
%
% mix_str           : estimated output mixture structure
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

[F, N, M, M] = size(mix_str_inp.Cx);

log_prior_small_const = 1e-70;

if nargin < 2 || isempty(upd_noise_flag)
    upd_noise_flag = 0;
end;

if M ~= 1 && M ~= 2
    error('GEM_iteration function is now implemented only for M = 1 or M = 2');
end;

mix_str = mix_str_inp;

if M == 2
    [Vs, As, rank_part_ind] = retrieve_subsrc_params(mix_str);

    % cimpute sufficient statistics
    % ------------------------------------------

    [hat_Rxx, hat_Rxs, hat_Rss, hat_Ws, log_like] = comp_suff_stat_M2(mix_str.Cx, Vs, As, mix_str.Noise_PSD);

    % update spatial parameters
    % ------------------------------------------

    % update mixing coefficients
    [mix_str, As] = update_mix_coef(mix_str, hat_Rxs, hat_Rss, As, rank_part_ind);

    % update noise PSD if necessary
    if upd_noise_flag
        for f = 1:F
            mix_str.Noise_PSD(f) = 0.5 * real(trace(squeeze(hat_Rxx(f,:,:)) - ...
                squeeze(As(:,f,:)) * squeeze(hat_Rxs(f,:,:))' - ...
                squeeze(hat_Rxs(f,:,:)) * squeeze(As(:,f,:))' + ...
                squeeze(As(:,f,:)) * squeeze(hat_Rss(f,:,:)) * squeeze(As(:,f,:))'));
        end;
    end;

    % squeeze spectral power suffitient statistics
    hat_W = squeeze_spec_power_suff_stat(hat_Ws, rank_part_ind);

    clear('Vs', 'As', 'rank_part_ind', 'hat_Rxx', 'hat_Rxs', 'hat_Rss', 'hat_Ws');
elseif M == 1
    % cimpute sufficient statistics
    % ------------------------------------------

    [hat_W, log_like] = comp_suff_stat_M1(mix_str);
else
    error('GEM_iteration function is now implemented only for M = 1 or M = 2');
end;

% add log-prior to log-likelihood for state-based models
for j = 1:length(mix_str.spec_comps)
    for l = 1:length(mix_str.spec_comps{j}.factors)
        factor_str = mix_str.spec_comps{j}.factors{l};
        
        if ~strcmp(factor_str.TW_constr, 'NMF') && isfield(factor_str, 'TW_DP_params')
            K = size(factor_str.TW_all, 1);
    
            [dummy, st_seq_ind] = max(factor_str.TW, [], 1);

            log_prior = 0;
            
            if strcmp(factor_str.TW_constr, 'GMM') || strcmp(factor_str.TW_constr, 'GSMM')
                for k = 1:K
                    log_prior = log_prior + numel(find(st_seq_ind == k)) * log(factor_str.TW_DP_params(k) + log_prior_small_const);
                end;
            elseif strcmp(factor_str.TW_constr, 'HMM') || strcmp(factor_str.TW_constr, 'SHMM')
                for k1 = 1:K
                    for k2 = 1:K
                        log_prior = log_prior + numel(find((st_seq_ind(1:N-1) == k1) & (st_seq_ind(2:N) == k2))) * log(factor_str.TW_DP_params(k1,k2) + log_prior_small_const);
                    end;
                end;
            else
                error('Unknown discrete state-based time weight constraint: %s', factor_str.TW_constr);
            end;
            
            log_like = log_like + log_prior / (F * N);
        end;
    end;
end;


% update spectral parameters
% ------------------------------------------
mix_str = update_spectral_components(mix_str, hat_W);

clear('hat_W');

% renormalize parameters
mix_str = renormalize_parameters(mix_str);
