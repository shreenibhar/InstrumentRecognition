function [mix_str, log_like_arr] = estim_param_a_post_model(mix_str_inp, ...
    iter_num, sim_ann_opt, Ann_PSD_beg, Ann_PSD_end)

%
% [mix_str, log_like_arr] = estim_param_a_post_model(mix_str_inp, ...
%    iter_num, sim_ann_opt, Ann_PSD_beg, Ann_PSD_end);
%
% estimate a posteriori mixture model parameters
%
%
% input
% -----
%
% mix_str_inp       : input mixture structure
% iter_num          : (opt) number of EM iterations (def = 100)
% sim_ann_opt       : (opt) simulated annealing option (def = 'ann')
%                        'no_ann'     : no annealing (zero noise)
%                        'ann'        : annealing
%                        'ann_ns_inj' : annealing with noise injection
%                        'upd_ns_prm' : update noise parameters
%                                       (Noise_PSD is updated through EM)
% Ann_PSD_beg       : (opt) [F x 1] beginning vector of annealing noise PSD
%                           (def = X_power / 100)
% Ann_PSD_end       : (opt) [F x 1] end vector of annealing noise PSD
%                           (def = X_power / 10000)
% 
%
% output
% ------
%
% mix_str           : estimated output mixture structure
% log_like_arr      : array of log-likelihoods
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

small_const = 1e-50;               % a small constant

if nargin < 2 || isempty(iter_num)
    iter_num = 100;
end;

if nargin < 3 || isempty(sim_ann_opt)
    sim_ann_opt = 'ann';
end;

[F, N, M, M] = size(mix_str_inp.Cx);

sig_pow = zeros(F, N);
for m = 1:M
    sig_pow = sig_pow + mix_str_inp.Cx(:,:,m,m) / M;
end;

Mix_PSD = mean(sig_pow, 2);

if nargin < 4 || isempty(Ann_PSD_beg)
    Ann_PSD_beg = Mix_PSD / 100;
end;

if nargin < 5 || isempty(Ann_PSD_end)
    Ann_PSD_end = Mix_PSD / 10000;
end;


if ~strcmp(sim_ann_opt, 'no_ann') && ~strcmp(sim_ann_opt, 'ann') && ...
   ~strcmp(sim_ann_opt, 'ann_ns_inj') && ~strcmp(sim_ann_opt, 'upd_ns_prm')
    error('Unknown sim_ann_opt');
end;


log_like_arr = ones(1, iter_num);


mix_str = mix_str_inp;


% adjust global enegry to mix
mix_str = adjust_glob_enegry_to_mix(mix_str);


if strcmp(sim_ann_opt, 'ann_ns_inj')
    Cx = mix_str.Cx;
end;

if strcmp(sim_ann_opt, 'upd_ns_prm')
    upd_noise_flag = 1;
else
    upd_noise_flag = 0;
end;

if strcmp(sim_ann_opt, 'ann') || strcmp(sim_ann_opt, 'ann_ns_inj') || strcmp(sim_ann_opt, 'upd_ns_prm')
    mix_str.Noise_PSD = Ann_PSD_beg;
elseif strcmp(sim_ann_opt, 'no_ann')
%    mix_str.Noise_PSD = ones(F, 1) * small_const;   % TO PREVENT from numerical errors
    mix_str.Noise_PSD = Ann_PSD_end;          % TO PREVENT from numerical errors
end;

% % spetial init for GSMM
% for spec_comp_ind = 1:length(mix_str.spec_comps)    
%     for l_ind = 1:length(mix_str.spec_comps{spec_comp_ind}.factors)
%         factor_str = mix_str.spec_comps{spec_comp_ind}.factors{l_ind};
% 
%         if strcmp(factor_str.TW_constr, 'GSMM')
%             [K, N] = size(factor_str.TW);
% 
%             if ~isfield(factor_str, 'TW_all')
%                 factor_str.TW_all = ones(K, 1) * max(factor_str.TW,[],1);
%             end;
%             
%             if numel(find(factor_str.TW ~= 0)) > N
%                 factor_str.TW = zeros(size(factor_str.TW));
%                 factor_str.TW(1,:) = factor_str.TW_all(1,:);
%             end;
%         end;
% 
%         mix_str.spec_comps{spec_comp_ind}.factors{l_ind} = factor_str;
%     end;
% end;

% MAIN LOOP
for iter = 1:iter_num
    % fprintf('GEM iteration %d of %d\n', iter, iter_num);

    % compute simulated annealing PSD (if necessary)
    if strcmp(sim_ann_opt, 'ann') || strcmp(sim_ann_opt, 'ann_ns_inj')
        Ann_PSD = ((sqrt(Ann_PSD_beg) * (iter_num - iter) + sqrt(Ann_PSD_end) * iter) / iter_num).^2;
        mix_str.Noise_PSD = Ann_PSD;
    end;

    if strcmp(sim_ann_opt, 'ann_ns_inj')
        Noise = randn(F, N, M) + sqrt(-1) * randn(F, N, M);
        
        Noise = Noise .* sqrt(repmat(Ann_PSD, [1, N, M]) / 2);
        
        for m1 = 1:M
            for m2 = 1:M
                mix_str.Cx(:,:,m1,m2) = Cx(:,:,m1,m2) + Noise(:,:,m1) .* conj(Noise(:,:,m2));
            end;
        end;
    end;

    [mix_str, log_like] = GEM_iteration(mix_str, upd_noise_flag);

    % if noise PSD is updated, apply a lower bound to it
    if upd_noise_flag
        mix_str.Noise_PSD = max(mix_str.Noise_PSD, Ann_PSD_end);
    end;

    if iter > 1
        log_like_diff = log_like - log_like_arr(iter-1);
        % fprintf('Log-likelihood: %f   Log-likelihood improvement: %f\n', log_like, log_like_diff);
    else
        % fprintf('Log-likelihood: %f\n', log_like);
    end;
    log_like_arr(iter) = log_like;
end;

if strcmp(sim_ann_opt, 'ann_ns_inj')
    mix_str.Cx = Cx;
end;
