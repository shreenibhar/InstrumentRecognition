function mix_str = adjust_glob_enegry_to_mix(mix_str_inp)

%
% mix_str = adjust_glob_enegry_to_mix(mix_str_inp);
%
% adjust global enegry to mix (for better initialization)
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
% mix_str           : estimated output mixture structure
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

M = size(mix_str_inp.Cx, 3);

data_power = 0;
for m = 1:M
    data_power = data_power + mean(mean(mix_str_inp.Cx(:,:,m,m))) / M;
end;

for spat_comp_ind = 1:length(mix_str.spat_comps)
    spat_comp_str = mix_str.spat_comps{spat_comp_ind};
    
    model_power = mean(abs(spat_comp_str.params(:)).^2);

    spat_comp_power = comp_spat_comp_power(mix_str, spat_comp_ind);
    model_power = model_power * mean(spat_comp_power(:));

    % scale spatial parameters
    spat_comp_str.params = spat_comp_str.params * sqrt(data_power / model_power);
    
    mix_str.spat_comps{spat_comp_ind} = spat_comp_str;
end;
