function mix_str = update_spectral_components(mix_str_inp, hat_W)

%
% mix_str = update_spectral_components(mix_str_inp, hat_W);
%
% update spectral component
%
%
% input
% -----
%
% inp_mix_str       : input mixture structure
% hat_W             : [F, N, K_spat] expected S-spectral power suffitient statistics
% 
%
% output
% ------
%
% mix_str           : updated output mixture structure
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

log_prior_small_const = 1e-70;

mix_str = mix_str_inp;

for spec_comp_ind = 1:length(mix_str.spec_comps)    
    factors_num = length(mix_str.spec_comps{spec_comp_ind}.factors);
    
    spat_comp_ind = mix_str.spec_comps{spec_comp_ind}.spat_comp_ind;

    for l_ind = 1:factors_num
        factor_str = mix_str.spec_comps{spec_comp_ind}.factors{l_ind};

        if strcmp(factor_str.FW_frdm_prior, 'part_fixed') || strcmp(factor_str.TW_frdm_prior, 'part_fixed') || strcmp(factor_str.TB_frdm_prior, 'part_fixed')
            error('part_fixed option is not yet implemented for FW, TW and TB');
        end;
        
        E_loc = ones(size(hat_W(:,:,spat_comp_ind)));
        if factors_num > 1
            E_loc = max(comp_spat_comp_power(mix_str, spat_comp_ind, spec_comp_ind, setdiff(1:factors_num, l_ind)), small_const); 
        end;
        
        % update FB
        if strcmp(factor_str.FB_frdm_prior, 'free') || strcmp(factor_str.FB_frdm_prior, 'part_fixed')
            V_loc = max(comp_spat_comp_power(mix_str, spat_comp_ind), small_const);

            comp_num = V_loc.^(-2) .* hat_W(:,:,spat_comp_ind) .* E_loc;
            comp_denum = V_loc.^(-1) .* E_loc;

            if isempty(factor_str.TB)
                H = factor_str.TW;
            else
                H = factor_str.TW * factor_str.TB;
            end;

            FW_H = factor_str.FW * H;

            comp_num = comp_num * FW_H';
            comp_denum = comp_denum * FW_H';

            FB_new = factor_str.FB .* comp_num ./ max(comp_denum, small_const);

            if strcmp(factor_str.FB_frdm_prior, 'free')
                factor_str.FB = FB_new;
            else
                factor_str.FB(:,factor_str.FB_free_col_inds) = FB_new(:,factor_str.FB_free_col_inds);
            end;

            mix_str.spec_comps{spec_comp_ind}.factors{l_ind} = factor_str;
        end;

        % update FW
        if strcmp(factor_str.FW_frdm_prior, 'free')
            V_loc = max(comp_spat_comp_power(mix_str, spat_comp_ind), small_const);

            comp_num = V_loc.^(-2) .* hat_W(:,:,spat_comp_ind) .* E_loc;
            comp_denum = V_loc.^(-1) .* E_loc;

            if isempty(factor_str.TB)
                H = factor_str.TW;
            else
                H = factor_str.TW * factor_str.TB;
            end;

            comp_num = factor_str.FB' * comp_num * H';
            comp_denum = factor_str.FB' * comp_denum * H';

%            if isempty(factor_str.FW_tied_inds)
                factor_str.FW = factor_str.FW .* comp_num ./ max(comp_denum, small_const);
%             else
%                 FW_new = comp_num ./ max(comp_denum, small_const);
% 
%                 for i = 1:length(factor_str.FW_tied_inds)
%                     cur_inds = factor_str.FW_tied_inds{i};
% 
%                     FW_new(cur_inds) = sum(comp_num(cur_inds)) / max(comp_denum(cur_inds), small_const);
%                 end;
% 
%                 factor_str.FW = FW_new;
%             end;

            mix_str.spec_comps{spec_comp_ind}.factors{l_ind} = factor_str;
        end;

        % update TW
        if strcmp(factor_str.TW_frdm_prior, 'free')

            if ~strcmp(factor_str.TW_constr, 'NMF') % processing for times weights with discrete state-based constrants
                [K, N] = size(factor_str.TW);
                
                if ~isempty(factor_str.TB)
                    error('In this implementation using nontrivial time blobs is incompatible with discrete state-based constrants, make sure that TW = [].');
                end;
                
                if ~isfield(factor_str, 'TW_all')
                    factor_str.TW_all = ones(K, 1) * max(factor_str.TW,[],1);
                end;
                
                if ~isfield(factor_str, 'TW_DP_params')
                    if strcmp(factor_str.TW_constr, 'GMM') || strcmp(factor_str.TW_constr, 'GSMM')
                        factor_str.TW_DP_params = ones(1, K) / K;
                    else
                        factor_str.TW_DP_params = ones(K, K) / K;
                    end;
                end;

                if (strcmp(factor_str.TW_constr, 'GMM') || strcmp(factor_str.TW_constr, 'HMM')) && (max(factor_str.TW_all(:)) > 1 || min(factor_str.TW_all(:)) < 1)
                    % transfer total energy to frequancy blobs
                    factor_str.FB = factor_str.FB * mean(factor_str.TW_all(:));

                    factor_str.TW_all(:) = 1;
                end;
                
                IS_dievergences = zeros(K, N);
                
                for k = 1:K
                    factor_str.TW(:)   = 0;
                    factor_str.TW(k,:) = factor_str.TW_all(k,:);
                    
                    mix_str.spec_comps{spec_comp_ind}.factors{l_ind} = factor_str;

                    if ~strcmp(factor_str.TW_constr, 'GMM') && ~strcmp(factor_str.TW_constr, 'HMM')
                        V_loc = max(comp_spat_comp_power(mix_str, spat_comp_ind), small_const);

                        comp_num = V_loc.^(-2) .* hat_W(:,:,spat_comp_ind) .* E_loc;
                        comp_denum = V_loc.^(-1) .* E_loc;

                        W = factor_str.FB * factor_str.FW;

                        comp_num = W' * comp_num;
                        comp_denum = W' * comp_denum;

                        factor_str.TW = factor_str.TW .* comp_num ./ max(comp_denum, small_const);

                        factor_str.TW_all(k,:) = factor_str.TW(k,:);

                        mix_str.spec_comps{spec_comp_ind}.factors{l_ind} = factor_str;
                    end;
                                        
                    % compute IS divergences
                    W_V_ratio = hat_W(:,:,spat_comp_ind) ./ max(comp_spat_comp_power(mix_str, spat_comp_ind), small_const);

                    IS_dievergences(k, :) = sum(W_V_ratio - log(max(W_V_ratio, small_const)) - 1, 1);
                end;
                
                % decode state sequence
                if strcmp(factor_str.TW_constr, 'GMM') || strcmp(factor_str.TW_constr, 'GSMM')
                    [dummy, st_seq_ind]  = min(IS_dievergences - log(factor_str.TW_DP_params + log_prior_small_const)' * ones(1,N), [], 1);
                elseif strcmp(factor_str.TW_constr, 'HMM') || strcmp(factor_str.TW_constr, 'SHMM')
                    % Viterbi algorithm
                    init_distr = ones(K, 1) / K;
                    
                    st_seq_ind = zeros(1, N);
                    
                    phi = IS_dievergences(:, 1) - log(init_distr);
                    psi = zeros(K, N);

                    % recurtion
                    for n = 2:N
                       [min_val, psi(:,n)] = min(ones(K,1) * phi' - log(factor_str.TW_DP_params + log_prior_small_const)', [], 2);
                       phi = min_val + IS_dievergences(:, n);
                    end;

                    % termination
                    [dummy, st_seq_ind(N)] = min(phi);

                    % path backtracking
                    for n = N-1:-1:1
                       st_seq_ind(n) = psi(st_seq_ind(n+1),n+1);
                    end;
                else
                    error('Unknown discrete state-based time weight constraint: %s', factor_str.TW_constr);
                end;

                % update time weights
                for n = 1:N
                    TW_n      = zeros(K, 1);
                    TW_n(st_seq_ind(n)) = factor_str.TW_all(st_seq_ind(n),n);
                    factor_str.TW(:,n) = TW_n;
                end;

                % update transition probabilities
                if strcmp(factor_str.TW_DP_frdm_prior, 'free')
                    if strcmp(factor_str.TW_constr, 'GMM') || strcmp(factor_str.TW_constr, 'GSMM')
                        for k = 1:K
                            factor_str.TW_DP_params(k) = numel(find(st_seq_ind == k)) / N;
                        end;
                    elseif strcmp(factor_str.TW_constr, 'HMM') || strcmp(factor_str.TW_constr, 'SHMM')
                        for k1 = 1:K
                            upd_denum = numel(find(st_seq_ind(1:N-1) == k1));
                            
                            if upd_denum > 0
                                for k2 = 1:K
                                    factor_str.TW_DP_params(k1,k2) = numel(find((st_seq_ind(1:N-1) == k1) & (st_seq_ind(2:N) == k2))) / upd_denum;
                                end;
                            end;
                        end;
                    else
                        error('Unknown discrete state-based time weight constraint: %s', factor_str.TW_constr);
                    end;
                end;
                
                mix_str.spec_comps{spec_comp_ind}.factors{l_ind} = factor_str;
            else   % processing for pure NMF
                V_loc = max(comp_spat_comp_power(mix_str, spat_comp_ind), small_const);

                comp_num = V_loc.^(-2) .* hat_W(:,:,spat_comp_ind) .* E_loc;
                comp_denum = V_loc.^(-1) .* E_loc;

                if isempty(factor_str.TB)
                    TB = 1;
                else
                    TB = factor_str.TB;
                end;

                W = factor_str.FB * factor_str.FW;

                comp_num = W' * comp_num * TB';
                comp_denum = W' * comp_denum * TB';

                factor_str.TW = factor_str.TW .* comp_num ./ max(comp_denum, small_const);

                mix_str.spec_comps{spec_comp_ind}.factors{l_ind} = factor_str;
            end;
        end;

        % update TB
        if ~isempty(factor_str.TB) && strcmp(factor_str.TB_frdm_prior, 'free')
            V_loc = max(comp_spat_comp_power(mix_str, spat_comp_ind), small_const);

            comp_num = V_loc.^(-2) .* hat_W(:,:,spat_comp_ind) .* E_loc;
            comp_denum = V_loc.^(-1) .* E_loc;

            W = factor_str.FB * factor_str.FW;

            W_TW = W * factor_str.TW;

            comp_num = W_TW' * comp_num;
            comp_denum = W_TW' * comp_denum;

            factor_str.TB = factor_str.TB .* comp_num ./ max(comp_denum, small_const);

            mix_str.spec_comps{spec_comp_ind}.factors{l_ind} = factor_str;
        end;  
    end;
    
%     % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%     IS_dievergence = 0;
% 
%     W_V_ratio = hat_W(:,:,spat_comp_ind) ./ max(comp_spat_comp_power(mix_str, spat_comp_ind), small_const);
% 
%     IS_dievergence = IS_dievergence + sum(sum(W_V_ratio - log(max(W_V_ratio, small_const)) - 1));
% 
%     IS_dievergence_after = IS_dievergence
%     % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
end;
