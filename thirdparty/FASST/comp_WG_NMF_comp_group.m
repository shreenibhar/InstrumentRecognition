function WG = comp_WG_NMF_comp_group(mix_str, NMF_ind_arr, factor_ind)

%
% WG = comp_WG_spec_comps(mix_str, NMF_ind_arr, factor_ind);
%
% compute Wiener gain for a group of NMF components
%
%
% input
% -----
%
% mix_str           : input mix structure
% NMF_ind_arr       : cell array of separated NMF components flags (1 = to extract)
% factor_ind         : factor index
% 
%
% output
% ------
%
% WG                : Wiener gains [M x M x F x N]
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

K_spat = length(mix_str.spat_comps);
K_spec = length(mix_str.spec_comps);

CLMN = ones(1,N);

Sigma_c_nmf = zeros(F,N,M,M);

Sigma_x = zeros(F,N,M,M);

WG = zeros(M,M,F,N);

if M == 1
    Sigma_x = mix_str.Noise_PSD * CLMN;

    for j = 1:K_spec
        spec_comp = mix_str.spec_comps{j};
        
        spat_comp = mix_str.spat_comps{spec_comp.spat_comp_ind};

        V = comp_spat_comp_power(mix_str, spec_comp.spat_comp_ind, j);

        A = spat_comp.params;
        if strcmp(spat_comp.mix_type, 'inst')
            A = repmat(A, [1, 1, F]);
        end;

        rank = size(A, 2);

        R = zeros(F,1);
        for r = 1:rank
            R = R + abs(squeeze(A(1,r,:))).^2;
        end;

        Sigma_x = Sigma_x + (R * CLMN) .* V;

        V_nmf = ones(F, N);

        for l_ind = 1:length(spec_comp.factors)
            factor_str = spec_comp.factors{l_ind};

            K_nmf = size(factor_str.FW, 2);

            if l_ind ~= factor_ind
                NMF_inds = 1:K_nmf;
            else
                NMF_inds = find(NMF_ind_arr{j});
            end;

            if isempty(factor_str.TB)
                H = factor_str.TW;
            else
                H = factor_str.TW * factor_str.TB;
            end;

            V_nmf = V_nmf .* (factor_str.FB * factor_str.FW(:,NMF_inds) * H(NMF_inds,:));
        end;

        Sigma_c_nmf = Sigma_c_nmf + (R * CLMN) .* V_nmf;
    end;
    
    % compute Wiener gain
    WG(1,1,:,:) = Sigma_c_nmf ./ Sigma_x;
elseif M == 2
    Sigma_x(:,:,1,1) = mix_str.Noise_PSD * CLMN;
    Sigma_x(:,:,2,2) = mix_str.Noise_PSD * CLMN;

    for j = 1:K_spec
        spec_comp = mix_str.spec_comps{j};
        
        spat_comp = mix_str.spat_comps{spec_comp.spat_comp_ind};

        V = comp_spat_comp_power(mix_str, spec_comp.spat_comp_ind, j);

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

        Sigma_x = Sigma_x + Sigma_c_loc;

        V_nmf = ones(F, N);

        for l_ind = 1:length(spec_comp.factors)
            factor_str = spec_comp.factors{l_ind};

            K_nmf = size(factor_str.FW, 2);

            if l_ind ~= factor_ind
                NMF_inds = 1:K_nmf;
            else
                NMF_inds = find(NMF_ind_arr{j});
            end;

            if isempty(factor_str.TB)
                H = factor_str.TW;
            else
                H = factor_str.TW * factor_str.TB;
            end;

            V_nmf = V_nmf .* (factor_str.FB * factor_str.FW(:,NMF_inds) * H(NMF_inds,:));
        end;

        Sigma_c_nmf(:,:,1,1) = Sigma_c_nmf(:,:,1,1) + (squeeze(R(1,1,:)) * CLMN) .* V_nmf;
        Sigma_c_nmf(:,:,1,2) = Sigma_c_nmf(:,:,1,2) + (squeeze(R(1,2,:)) * CLMN) .* V_nmf;
        Sigma_c_nmf(:,:,2,2) = Sigma_c_nmf(:,:,2,2) + (squeeze(R(2,2,:)) * CLMN) .* V_nmf;
        Sigma_c_nmf(:,:,2,1) = conj(Sigma_c_nmf(:,:,1,2));
    end;

    % compute Sigma_x matrix inverse
    Inv_Sigma_x = inv_herm_matr_2D(Sigma_x);

    clear('Sigma_x');

    % compute Wiener gain
    WG = permute(mult_matr_FN(Sigma_c_nmf, Inv_Sigma_x), [3 4 1 2]);
else
    error('comp_suff_stat_M2 function is only implemented for M = 1 and M = 2');   
end;
