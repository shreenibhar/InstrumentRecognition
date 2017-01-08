function V = comp_spat_comp_power(mix_str, spat_comp_ind, spec_comp_ind, factor_ind)

%
% V = comp_spat_comp_power(mix_str, spat_comp_ind, spec_comp_ind, factor_ind);
%
% compute spatial component power
%
%
% input
% -----
%
% mix_str           : mixture structure
% spat_comp_ind     : spatial component index
% spec_comp_ind     : (opt) factor index (def = [], use all components)
% factor_ind         : (opt) factor index (def = [], use all factors)
% 
%
% output
% ------
%
% V                 : (F x N) spatial component power
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

if nargin < 3 || isempty(spec_comp_ind)
    spec_comp_ind = [];
end;

if nargin < 4 || isempty(factor_ind)
    factor_ind = [];
end;

[F, N, M, M] = size(mix_str.Cx);

V = zeros(F, N);

spec_comp_ind_arr = 1:length(mix_str.spec_comps);
if ~isempty(spec_comp_ind)
    spec_comp_ind_arr = spec_comp_ind;
end;

for j = spec_comp_ind_arr
    if spat_comp_ind == mix_str.spec_comps{j}.spat_comp_ind
        V_comp = ones(F, N);

        factors_ind_arr = 1:length(mix_str.spec_comps{j}.factors);
        if ~isempty(factor_ind)
            factors_ind_arr = factor_ind;
        end;
        
        for factor_ind_cur = factors_ind_arr
            factor_str = mix_str.spec_comps{j}.factors{factor_ind_cur};

            W = factor_str.FB * factor_str.FW;

            if isempty(factor_str.TB)
                H = factor_str.TW;
            else
                H = factor_str.TW * factor_str.TB;
            end;

            V_comp = V_comp .* (W * H);
        end;

        V = V + V_comp;
    end;
end;
