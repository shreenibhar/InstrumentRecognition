function WG = comp_WG_spec_comps(mix_str, sep_cmp_inds)

%
% WG = comp_WG_spec_comps(mix_str, sep_cmp_inds);
%
% compute Wiener gains for spectral components
%
%
% input
% -----
%
% mix_str           : input mix structure
% sep_cmp_inds      : (opt) array of indices for components to separate
%                     (def = {1, 2, ..., K_spec})
% 
%
% output
% ------
%
% WG                : Wiener gains [M x M x F x N x K_sep],
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

if nargin < 2 || isempty(sep_cmp_inds)
    sep_cmp_inds = cell(1, K_spec);
    for j = 1:K_spec
        sep_cmp_inds{j} = j;
    end;
end;

[F, N, M, M] = size(mix_str.Cx);

K_sep = length(sep_cmp_inds);


CLMN = ones(1,N);

Sigma_c = zeros(K_sep,F,N,M,M);

Sigma_x = zeros(F,N,M,M);

WG = zeros(M,M,F,N,K_sep);

if M == 1
    Sigma_x = mix_str.Noise_PSD * CLMN;

    for m = 1:K_sep
        for j = sep_cmp_inds{m}
            k = mix_str.spec_comps{j}.spat_comp_ind;

            spat_comp = mix_str.spat_comps{k};

            V = comp_spat_comp_power(mix_str, k, j);

            A = spat_comp.params;
            if strcmp(spat_comp.mix_type, 'inst')
                A = repmat(A, [1, 1, F]);
            end;

            rank = size(A, 2);

            R = zeros(F,1);
            for r = 1:rank
                R = R + abs(squeeze(A(1,r,:))).^2;
            end;

            Sigma_c_loc = (R * CLMN) .* V;

            Sigma_c(m,:,:) = squeeze(Sigma_c(m,:,:)) + Sigma_c_loc;

            Sigma_x = Sigma_x + Sigma_c_loc;
        end;
    end;
    
    for m = 1:K_sep
        % compute Wiener gain
        WG(1,1,:,:,m) = squeeze(Sigma_c(m,:,:)) ./ Sigma_x;
    end;
elseif M == 2
    Sigma_x(:,:,1,1) = mix_str.Noise_PSD * CLMN;
    Sigma_x(:,:,2,2) = mix_str.Noise_PSD * CLMN;

    for m = 1:K_sep
        for j = sep_cmp_inds{m}
            k = mix_str.spec_comps{j}.spat_comp_ind;

            spat_comp = mix_str.spat_comps{k};

            V = comp_spat_comp_power(mix_str, k, j);

            A = spat_comp.params;
            if strcmp(spat_comp.mix_type, 'inst')
                A = repmat(A, [1, 1, F]);
            end;

            rank = size(A, 2);

            R = zeros(M,M,F);
            for r = 1:rank
                R(1,1,:) = R(1,1,:) + abs(A(1,r,:)).^2;
                R(1,2,:) = R(1,2,:) + A(1,r,:) .* conj(A(2,r,:));
                R(2,2,:) = R(2,2,:) + abs(A(2,r,:)).^2;
                R(2,1,:) = R(2,1,:) + conj(A(1,r,:) .* conj(A(2,r,:)));
            end;

            Sigma_c_loc = zeros(F,N,M,M);
            Sigma_c_loc(:,:,1,1) = (squeeze(R(1,1,:)) * CLMN) .* V;
            Sigma_c_loc(:,:,1,2) = (squeeze(R(1,2,:)) * CLMN) .* V;
            Sigma_c_loc(:,:,2,2) = (squeeze(R(2,2,:)) * CLMN) .* V;
            Sigma_c_loc(:,:,2,1) = conj(Sigma_c_loc(:,:,1,2));

            Sigma_c(m,:,:,:,:) = squeeze(Sigma_c(m,:,:,:,:)) + Sigma_c_loc;

            Sigma_x = Sigma_x + Sigma_c_loc;
        end;
    end;

    % compute Sigma_x matrix inverse
    Inv_Sigma_x = inv_herm_matr_2D(Sigma_x);

    clear('Sigma_x');

    for m = 1:K_sep
        % compute Wiener gain
        WG(:,:,:,:,m) = permute(mult_matr_FN(squeeze(Sigma_c(m,:,:,:,:)), Inv_Sigma_x), [3 4 1 2]);
    end;
else
    error('comp_suff_stat_M2 function is only implemented for M = 1 and M = 2');   
end;
