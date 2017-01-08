function [hat_Rxx, hat_Rxs, hat_Rss, hat_Ws, log_like] = comp_suff_stat_M2(Cx, Vs, As, Noise_PSD)

%
% [hat_Rxx, hat_Rxs, hat_Rss, hat_Ws, log_like] = comp_suff_stat_M2(Cx, Vs, As, Noise_PSD);
%
% compute suffitient statistics for GEM algorithm in the 2 channels case
%
%
% input
% -----
%
% Cx                : [F x N x M x M] matrix containing the spatial covariance
%                     matrices of the input signal in all time-frequency bins
%                     or [F x N] single channel variance matrix
% Vs                : [F, N, R] spectral power of point sub-sources
% As                : [M, F, R] mixing coefficients of rank-1 sub-sources
% Noise_PSD         : [F X 1] vector of noise PSD
% 
%
% output
% ------
%
% hat_Rxx           : [F,2,2] expected Rxx suffitient statistics
% hat_Rxs           : [F,2,R] expected Rxs suffitient statistics
% hat_Rss           : [F,R,R] expected Rss suffitient statistics
% hat_Ws            : [F,N,R] expected S-spectral power suffitient statistics
% log_like          : log-likelihood
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

[F, N, M, M] = size(Cx);
R = size(Vs, 3);

if M ~= 2
    error('comp_suff_stat_M2 function is only implemented for M = 2');
end;

CLMN = ones(1,N);

% compute the Sigma_x matrix
Sigma_x = zeros(F,N,2,2);

Sigma_x(:,:,1,1) = Noise_PSD * CLMN;
Sigma_x(:,:,2,2) = Noise_PSD * CLMN;
for r = 1:R
    Sigma_x(:,:,1,1) = Sigma_x(:,:,1,1) + (abs(squeeze(As(1,:,r))).^2.' * CLMN) .* Vs(:,:,r);
    Sigma_x(:,:,1,2) = Sigma_x(:,:,1,2) + ((squeeze(As(1,:,r)).' .* conj(squeeze(As(2,:,r))).') * CLMN) .* Vs(:,:,r);
    Sigma_x(:,:,2,2) = Sigma_x(:,:,2,2) + (abs(squeeze(As(2,:,r))).^2.' * CLMN) .* Vs(:,:,r);
end;
Sigma_x(:,:,2,1) = conj(Sigma_x(:,:,1,2));


% compute Sigma_x matrix inverse
[Inv_Sigma_x, Det_Sigma_x] = inv_herm_matr_2D(Sigma_x);

clear('Sigma_x');

% compute log-likelihood
log_like = -sum(sum(log(Det_Sigma_x * pi) + ...
    Inv_Sigma_x(:,:,1,1) .* Cx(:,:,1,1) + ...
    Inv_Sigma_x(:,:,2,2) .* Cx(:,:,2,2) + ...
    2 * real(Inv_Sigma_x(:,:,1,2) .* Cx(:,:,2,1)) )) / (N * F);

% compute expectations of Rss and Ws suffitient statistics
% --------------------------------------------------------
Gs = zeros(F,N,R,2);

for r = 1:R
    % compute S-Wiener gain
    Gs(:,:,r,1) = ((conj(squeeze(As(1,:,r))).' * CLMN) .* Inv_Sigma_x(:,:,1,1) + ...
                   (conj(squeeze(As(2,:,r))).' * CLMN) .* Inv_Sigma_x(:,:,2,1)) .* Vs(:,:,r);
    Gs(:,:,r,2) = ((conj(squeeze(As(1,:,r))).' * CLMN) .* Inv_Sigma_x(:,:,1,2) + ...
                   (conj(squeeze(As(2,:,r))).' * CLMN) .* Inv_Sigma_x(:,:,2,2)) .* Vs(:,:,r);
end;

hat_Rss = zeros(F,R,R);
hat_Ws  = zeros(F,N,R);

for r1 = 1:R
    % compute average Rss
    for r2 = 1:R
        hat_Rss_local = Gs(:,:,r1,1) .* conj(Gs(:,:,r2,1)) .* Cx(:,:,1,1) + ...
                        Gs(:,:,r1,2) .* conj(Gs(:,:,r2,2)) .* Cx(:,:,2,2) + ...
                        Gs(:,:,r1,1) .* conj(Gs(:,:,r2,2)) .* Cx(:,:,1,2) + ...
                        Gs(:,:,r1,2) .* conj(Gs(:,:,r2,1)) .* Cx(:,:,2,1) ...
            - (Gs(:,:,r1,1) .* (squeeze(As(1,:,r2)).' * CLMN) + ...
               Gs(:,:,r1,2) .* (squeeze(As(2,:,r2)).' * CLMN)) .* Vs(:,:,r2);

        if r1 == r2
            hat_Rss_local = hat_Rss_local + Vs(:,:,r1);

            hat_Ws(:,:,r1) = abs(real(hat_Rss_local));
        end;

        hat_Rss(:,r1,r2) = mean(hat_Rss_local, 2);            
    end;
end;

% TO ASSURE that hat_Rss = hat_Rss'
for f = 1:F
    hat_Rss(f,:,:) = (squeeze(hat_Rss(f,:,:)) + squeeze(hat_Rss(f,:,:))') / 2;
end;

% compute expectations of Rxs suffitient statistics
% -------------------------------------------------

hat_Rxs = zeros(F,2,R);

for r = 1:R
    hat_Rxs(:,1,r) = mean(conj(Gs(:,:,r,1)) .* Cx(:,:,1,1) + ...
                          conj(Gs(:,:,r,2)) .* Cx(:,:,1,2), 2);

    hat_Rxs(:,2,r) = mean(conj(Gs(:,:,r,1)) .* Cx(:,:,2,1) + ...
                          conj(Gs(:,:,r,2)) .* Cx(:,:,2,2), 2);
end;

clear('Gs');

% compute expectations of Rxx suffitient statistics
% -------------------------------------------------

hat_Rxx = reshape(mean(Cx,2), [F, M, M]);

% TO ASSURE that Rxx = Rxx'
hat_Rxx(:,1,2) = squeeze((hat_Rxx(:,1,2) + conj(hat_Rxx(:,2,1))) / 2);
hat_Rxx(:,2,1) = squeeze(conj(hat_Rxx(:,1,2)));
hat_Rxx(:,1,1) = squeeze(real(hat_Rxx(:,1,1)));
hat_Rxx(:,2,2) = squeeze(real(hat_Rxx(:,2,2)));
