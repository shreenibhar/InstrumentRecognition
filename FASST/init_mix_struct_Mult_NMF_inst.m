function mix_str = init_mix_struct_Mult_NMF_inst(Cx, J, K, transf, fs, wlen)

%
% mix_str = init_mix_struct_Mult_NMF_inst(Cx, J, K, transf, fs, wlen);
%
% An example of mixture structure initialization, corresponding to
% multichannel NMF model (instantaneous case)
% Most of parameters are initialized randomly
%
% input 
% -----
%
% Cx                : [F x N x I x I] matrix containing the spatial covariance
%                     matrices of the input signal in all time-frequency bins
%                     or [F x N] single channel variance matrix
% J                 : number of components (here J_spat = J_spec)
% K                 : number of NMF components per source
% transf            : transform ('stft' or 'qerb')
% fs                : sampling frequency in Hz
% wlen              : length of the time integration window (must be a power of 2)
%
% output
% ------
%
% mix_str           : initialized mixture structure
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

rank = 1;

[F, N, I, I] = size(Cx);

mix_str.Cx         = Cx;
mix_str.transf     = transf;
mix_str.fs         = fs;
mix_str.wlen       = wlen;
mix_str.spat_comps = cell(1,J);
mix_str.spec_comps = cell(1,J);

for j = 1:J
    % initialize spatial component
    mix_str.spat_comps{j}.time_dep   = 'indep';
    mix_str.spat_comps{j}.mix_type   = 'inst';
    mix_str.spat_comps{j}.frdm_prior = 'free';
    mix_str.spat_comps{j}.params     = randn(I, rank);

    % initialize single factor spectral component
    mix_str.spec_comps{j}.spat_comp_ind = j;
    mix_str.spec_comps{j}.factors        = cell(1, 1);
    
    factor1.FB            = 0.75 * abs(randn(F, K)) + 0.25 * ones(F, K);
    factor1.FW            = diag(ones(1, K));
    factor1.TW            = 0.75 * abs(randn(K, N)) + 0.25 * ones(K, N);
    factor1.TB            = [];
    factor1.FB_frdm_prior = 'free';
    factor1.FW_frdm_prior = 'fixed';
    factor1.TW_frdm_prior = 'free';
    factor1.TB_frdm_prior = [];
    factor1.TW_constr     = 'NMF';
    
    mix_str.spec_comps{j}.factors{1} = factor1;
end;
