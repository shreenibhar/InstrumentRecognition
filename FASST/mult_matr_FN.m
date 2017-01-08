function C = mult_matr_FN(A, B)

%
% C = mult_matr_FN(A, B);
%
% multiply two matrices for every time-frequency point
%
%
% input
% -----
%
% A                 : [F,N,a1,a2] array of (a1 x a2) matrices
% B                 : [F,N,b1,b2] array of (b1 x b2) matrices
% 
%
% output
% ------
%
% C                 : [F,N,c1,c2] array of (c1 x c2) matrix products (A x B)
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

[F, N, a1, a2] = size(A);
[F, N, b1, b2] = size(B);

if a2 ~= b1
    error('mult_matr_FN function : incorrect matrix dimensions');
end;

C = zeros(F, N, a1, b2);

for i = 1:a1
    for j = 1:b2
        for k = 1:a2
            C(:,:,i,j) = C(:,:,i,j) + A(:,:,i,k) .* B(:,:,k,j);
        end;
    end;
end;
