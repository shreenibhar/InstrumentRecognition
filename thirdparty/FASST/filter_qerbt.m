function s=filter_qerbt(x,W,fs,wlen)

% FILTER_QERBT Multichannel filtering in the ERB-scale time-frequency
% domain
%
% s=filter_qerbt(x,W,fs,wlen)
%
% Inputs:
% x: nsampl x ichan vector containing a multichannel (e.g. mixture) signal
% W: ochan x ichan x nbin x nfram matrix containing multichannel filter
% coefficients (e.g. Wiener filter coefficients for all sources) for each
% time-frequency bin
% fs: sampling frequency in Hz
% wlen: length of the time integration window (must be a power of 2)
%
% Output:
% s: nsampl x ochan matrix containing the extracted (e.g. source) signals
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
[nsampl,ichan]=size(x);
if ichan>nsampl, error('The input signal must be in columns.'); end
[ochan,ichan2,nbin,nfram]=size(W);
if ichan~=ichan2, error('The filter must have the same number of channels as the input signal.'); end
nfram2=ceil(nsampl/wlen*2);
if nfram~=nfram2, error('The length of the time integration window must be consistent with the number of frames of the filter.'); end

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
% Determining filterbank and inverse filterbank magnitude response
ngrid=1000;
egrid=(0:ngrid-1)*emax/(ngrid-1);
fgrid=(exp(egrid/9.26)-1)/.00437;
resp=zeros(ngrid,nbin);
for bin=1:nbin,
    hwlen=round(a(bin)/subs(bin));
    alen=(2*hwlen+1)*subs(bin);
    r=alen/fs*(fgrid-f(bin));
    resp(:,bin)=(sinc(r)+.5*sinc(r+1)+.5*sinc(r-1)).^2;
end
wei=pinv(resp)*ones(ngrid,1);

%%% Filtering %%%
x=[x; zeros((nfram+1)*wlen/2-nsampl,ichan)];
for i=1:ichan,
    x(:,i)=hilbert(x(:,i));
end
% Defining the time integration window
win=sin((.5:wlen-.5)/wlen*pi).';
swin=zeros((nfram+1)*wlen/2,1);
for t=0:nfram-1,
    swin(t*wlen/2+1:t*wlen/2+wlen)=swin(t*wlen/2+1:t*wlen/2+wlen)+win.^2;
end
swin=sqrt(swin);
s=zeros((nfram+1)*wlen/2,ochan);
sscale=zeros((nfram+1)*wlen/2,ochan);
for bin=nbin:-1:1,
    % Dyadic downsampling/upsampling
    if down(bin),
        x=resample(x,1,2,50);
        wlen=wlen/2;
        win=sin((.5:wlen-.5)/wlen*pi).';
        swin=zeros((nfram+1)*wlen/2,1);
        for t=0:nfram-1,
            swin(t*wlen/2+1:t*wlen/2+wlen)=swin(t*wlen/2+1:t*wlen/2+wlen)+win.^2;
        end
        swin=sqrt(swin);
        s=s+resample(sscale,subs(bin+1),1,50);
        sscale=zeros((nfram+1)*wlen/2,ochan);
    end
    % Filterbank
    hwlen=round(a(bin)/subs(bin));
    filt=1/(hwlen+1)*hanning(2*hwlen+1).*exp(complex(0,1)*2*pi*f(bin)/fs*subs(bin)*(-hwlen:hwlen).');
    iband=fftfilt(filt,[x;zeros(2*hwlen,ichan)]);
    iband=iband(hwlen+1:hwlen+(nfram+1)*wlen/2,:);
    % Bandwise filtering
    oband=zeros((nfram+1)*wlen/2,ochan);
    for t=0:nfram-1,
        fram=iband(t*wlen/2+1:t*wlen/2+wlen,:).*repmat((win./swin(t*wlen/2+1:t*wlen/2+wlen)).^2,[1 ichan]);
        oband(t*wlen/2+1:t*wlen/2+wlen,:)=oband(t*wlen/2+1:t*wlen/2+wlen,:)+fram*W(:,:,bin,t+1).';
    end
    % Inverse filterbank
    oband=real(fftfilt(filt,[oband;zeros(2*hwlen,ochan)]));
    sscale=sscale+wei(bin)*oband(hwlen+1:hwlen+(nfram+1)*wlen/2,:);
end
s=s+resample(sscale,subs(1),1,50);
s=s(1:nsampl,:);

return;