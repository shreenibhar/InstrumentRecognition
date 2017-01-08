function s=filter_stft(x,W)

% FILTER_STFT Multichannel filtering in the STFT time-frequency
% domain
%
% s=filter_stft(x,W)
%
% Inputs:
% x: nsampl x ichan vector containing a multichannel (e.g. mixture) signal
% W: ochan x ichan x nbin x nfram matrix containing multichannel filter
% coefficients (e.g. Wiener filter coefficients for all sources) for each
% time-frequency bin
%
% Output:
% s: nsampl x ochan matrix containing the extracted (e.g. source) signals
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

[ochan, ichan, nbin, nfram] = size(W);

stft_win_len = 2*(nbin-1);

x = x.';
mix_nsamp = size(x,2);
X=stft_multi(x,stft_win_len);

Se = zeros(nbin, nfram, ochan);
for i = 1:ichan
    for j = 1:ochan
        Se(:, :, j) = Se(:, :, j) + squeeze(W(j, i, :, :)) .* X(:, :, i);
    end;
end;

s = istft_multi(Se,mix_nsamp);
s = s.';
