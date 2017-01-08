function [Cx,f]=qerbt(x,fs,nbin,wlen)

% QERBT Quadratic ERB-scale time-frequency transform using Hann filters and
% half-overlapping sine time integration windows
%
% [Cx,f]=qerbt(x,fs,nbin,wlen)
%
% Inputs:
% x: nsampl x nchan vector containing a multichannel signal
% fs: sampling frequency in Hz
% nbin: number of frequency bins
% wlen: length of the time integration window (must be a power of 2)
%
% Output:
% Cx: nchan x nchan x nbin x nfram matrix containing the spatial covariance
% matrices of the input signal in all time-frequency bins
% f: nbin x 1 vector containing the center frequency of each frequency bin
% in Hz
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2002-2010 Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Emmanuel Vincent, "Musical source separation using time-frequency source
% priors," IEEE Trans. on Audio, Speech and Language Processing,
% 14(1):91-98, 2006
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Errors and warnings %%%
if nargin<4, error('Not enough input arguments.'); end
[nsampl,nchan]=size(x);
if nchan>nsampl, error('The input signal must be in columns.'); end
nfram=ceil(nsampl/wlen*2);

%%% Defining ERB scale parameters %%%
% Determining frequency and window length scales
emax=9.26*log(.00437*fs/2+1);
e=(0:nbin-1)*emax/(nbin-1);
f=(exp(e/9.26)-1)/.00437;
a=.5*(nbin-1)/emax*9.26*.00437*fs*exp(-e/9.26)-.5;
% Determining dyadic downsampling factors (for fast computation)
fup=f+1.5*fs./(2*a+1);
subs=-log2(2*fup/fs);
subs=2.^max(0,floor(min(log2(wlen/2),subs)));
down=(subs~=[subs(2:end),1]);

%%% Computing QERBT coefficients %%%
x=[x; zeros((nfram+1)*wlen/2-nsampl,nchan)];
for i=1:nchan,
    x(:,i)=hilbert(x(:,i));
end
% Defining the time integration window
win=sin((.5:wlen-.5)/wlen*pi).';
swin=zeros((nfram+1)*wlen/2,1);
for t=0:nfram-1,
    swin(t*wlen/2+1:t*wlen/2+wlen)=swin(t*wlen/2+1:t*wlen/2+wlen)+win.^2;
end
swin=sqrt(swin);
Cx=zeros(nchan,nchan,nbin,nfram);
for bin=nbin:-1:1,
    % Dyadic downsampling
    if down(bin),
        x=resample(x,1,2,50);
        wlen=wlen/2;
        win=sin((.5:wlen-.5)/wlen*pi).';
        swin=zeros((nfram+1)*wlen/2,1);
        for t=0:nfram-1,
            swin(t*wlen/2+1:t*wlen/2+wlen)=swin(t*wlen/2+1:t*wlen/2+wlen)+win.^2;
        end
        swin=sqrt(swin);
    end
    % Filterbank
    hwlen=round(a(bin)/subs(bin));
    filt=hanning(2*hwlen+1).*exp(complex(0,1)*2*pi*f(bin)/fs*subs(bin)*(-hwlen:hwlen).');
    band=fftfilt(filt,[x;zeros(2*hwlen,nchan)]);
    band=band(hwlen+1:hwlen+(nfram+1)*wlen/2,:);
    % Time integration
    for t=0:nfram-1,
        fram=band(t*wlen/2+1:t*wlen/2+wlen,:).*repmat(win./swin(t*wlen/2+1:t*wlen/2+wlen),[1 nchan]);
        Cx(:,:,bin,t+1)=subs(bin)/(hwlen+1)^2*conj((fram'*fram));
    end
end

return;