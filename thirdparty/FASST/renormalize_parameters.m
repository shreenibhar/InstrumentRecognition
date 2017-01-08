function mix_str = renormalize_parameters(mix_str_inp)

%
% SAC_str = normalize_SAC_parameters(SAC_str_inp);
%
% update full rank spatial covariance matrices
%
%
% input
% -----
%
% mix_str_inp       : input mixture structure
% 
%
% output
% ------
%
% mix_str           : updated mixture structure
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

K_spat = length(mix_str.spat_comps);

% renormalize spatial components
spat_glob_energy = zeros(1, K_spat);
for spat_ind = 1:K_spat
    spat_glob_energy(spat_ind) = mean(abs(mix_str.spat_comps{spat_ind}.params(:)) .^ 2);

    mix_str.spat_comps{spat_ind}.params = mix_str.spat_comps{spat_ind}.params / sqrt(spat_glob_energy(spat_ind));
end;


K_spec = length(mix_str.spec_comps);

% renormalize spectral components
for spec_ind = 1:K_spec
    glob_energy = spat_glob_energy(mix_str.spec_comps{spec_ind}.spat_comp_ind);
        
    factors_num = length(mix_str.spec_comps{spec_ind}.factors);
    
    for l_ind = 1:factors_num
        factor_str = mix_str.spec_comps{spec_ind}.factors{l_ind};

        factor_str.FB = factor_str.FB * glob_energy;                        % Normamize FB by previous global energy

        w=mean(factor_str.FB, 1);  w(find(w == 0)) = 1;  d=diag(ones(size(w)) ./ w);
        factor_str.FB = factor_str.FB * d;                                  % Normalisation of FB
        d=diag(w);  factor_str.FW = d * factor_str.FW;                      % Energy transfer to FW

        if ~strcmp(factor_str.TW_constr, 'GMM') && ~strcmp(factor_str.TW_constr, 'HMM')
            w=mean(factor_str.FW, 1);  w(find(w == 0)) = 1;  d=diag(ones(size(w)) ./ w);
            factor_str.FW = factor_str.FW * d;                                  % Normalisation of FW
            d=diag(w);  factor_str.TW = d * factor_str.TW;                      % Energy transfer to TW

            if ~isempty(factor_str.TB)
                w=mean(factor_str.TB, 2);  w(find(w == 0)) = 1;  d=diag(ones(size(w)) ./ w);
                factor_str.TB = d * factor_str.TB;                              % Normalisation of TB
                d=diag(w);  factor_str.TW = factor_str.TW * d;                  % Energy transfer to TW
            end;

            glob_energy = mean(factor_str.TW(:));
            
            if l_ind < factors_num
                factor_str.TW = factor_str.TW / glob_energy;                    % Normamize TW by its global energy
            end;
        else % GMM or HMM
            if ~isempty(factor_str.TB)
                error('GMMs and HMMs are not compatible with nonempty time blobs');
            end;

            glob_energy = mean(factor_str.FW(:));
            
            if l_ind < factors_num
                factor_str.FW = factor_str.FW / glob_energy;                    % Normamize TW by its global energy
            end;
        end;

        mix_str.spec_comps{spec_ind}.factors{l_ind} = factor_str;
    end;
end;
