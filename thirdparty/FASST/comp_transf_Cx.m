function Cx = comp_transf_Cx(x, transf, win_len, fs, qerb_nbin)

%
% Cx = comp_transf_Cx(x, transf, win_len, fs, qerb_nbin);
%
% compute spatial covariance matrices for the corresponding transform
%
%
% input 
% -----
%
% x                 : [I x nsampl] matrix containing I time-domain mixture signals
%                     with nsampl samples
% transf            : transform
%                       'stft'
%                       'qerb'
% win_len           : window length
% fs                : (opt) sampling frequency (Hz)
% qerb_nbin         : (opt) number of bins for qerb transform
%
% output
% ------
%
% Cx                : [F x N x I x I] matrix containing the spatial covariance
%                     matrices of the input signal in all time-frequency bins
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


if ~strcmp(transf, 'stft') && ~strcmp(transf, 'qerb')
    error('Unknown transf');
end;


if strcmp(transf, 'stft')
    X=stft_multi(x, win_len);
    [F, N, I] = size(X);
    Cx = zeros(F, N, I, I);
    for m1=1:I
        for m2=1:I
            if m1 <= m2
                Cx(:, :, m1, m2) = X(:, :, m1) .* conj(X(:, :, m2));
            else
                Cx(:, :, m1, m2) = conj(Cx(:, :, m2, m1));
            end;
        end;
    end;
elseif strcmp(transf, 'qerb')
    Cx = qerbt(x.', fs, qerb_nbin, win_len);
    Cx = permute(Cx, [3 4 1 2]);
else
    error('Unknown transf');
end;
