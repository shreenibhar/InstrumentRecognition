function [Inv_A, Det_A] = inv_herm_matr_2D(A)

%
% [Inv_A, Det_A] = inv_herm_matr_2D(A);
%
% fast inversion of two dimentional Hermitian matrices using Cramer?s
%   explicit matrix inversion formula
%
%
% input
% -----
%
% A                 : [F,N,2,2] array of two dimentional Hermitian matrices
% 
%
% output
% ------
%
% Inv_A             : [F,N,2,2] array of inverted two dimentional Hermitian matrices
% Det_A             : [F,N] matrix of determinants of inverted A matrices
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

[F, N, M1, M2] = size(A);

if (M1 ~= 2) || (M2 ~= 2)
    error('inv_herm_matr_2D function is only implemented for two dimentional matrices');
end;

% compute the inverse of A matrix
Det_A = A(:,:,1,1) .* A(:,:,2,2) - abs(A(:,:,1,2)).^2;
Inv_A(:,:,1,1) =   A(:,:,2,2) ./ Det_A;
Inv_A(:,:,1,2) = - A(:,:,1,2) ./ Det_A;
Inv_A(:,:,2,1) =   conj(Inv_A(:,:,1,2));
Inv_A(:,:,2,2) =   A(:,:,1,1) ./ Det_A;
