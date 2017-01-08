function [V,B,harm_num]=spectral_blobs_erb(nbin,fs,hcomp,wcomp,firstnote,lastnote,hwidth,wwidth,eslope)

% SPECTRAL_BLOBS_ERB Defines the structure of harmonic and wideband basis
% spectra from basis blobs and (initial) spectral envelopes
%
% The ERB transform is assumed to be computed via a Hanning window
%
% Initialization of harmonic spectral envelopes with a fixed slope and
% random initialization of wideband spectral envelopes
%
% [V,B]=spectral_blobs_erb(nbin,fs,hcomp,wcomp,firstnote,lastnote,hwidth,wwidth,eslope)
%
% Inputs:
% nbin: number of ERB frequency bins
% fs: sampling frequency in Hz
% hcomp: number of harmonic components PER PITCH
% wcomp: number of wideband components
% firstnote: lowest pitch on the MIDI scale
% lastnote: highest pitch on the MIDI scale
% hwidth: harmonic blob width in ERB
% wwidth: wideband blob width in ERB
% eslope: initial spectral envelope slope in dB/oct for the first harmonic
% component (twice for the second, etc)
%
% Output:
% V: (wlen/2+1) x nblob matrix containing blob power spectra
% B: nblob x ncomp matrix containing power spectral envelopes
% harm_num: number of harmonic components
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

%%% Computing ERB scale %%%
emax=9.26*log(.00437*fs/2+1);
e=(0:nbin-1)*emax/(nbin-1);
a=.5*(nbin-1)/emax*9.26*.00437*fs*exp(-e/9.26)-.5;
alen=2*round(a)+1;
f=(exp(e/9.26)-1)/(.00437*fs);

%%% Defining the power spectra of sinusoidal partials %%%
nbnotes=lastnote-firstnote+1; pitch=firstnote:lastnote;
f0=2.^((pitch-69)/12)*440/fs;
nharm=floor(.5./f0);
ppos=[0,cumsum(nharm)];
nbpart=ppos(end);
Z=zeros(nbin,nbpart);   %partial spectra
partfreq=zeros(1,nbpart);
for n=1:nbnotes,
    partfreq(ppos(n)+1:ppos(n)+nharm(n))=f0(n)*(1:nharm(n));
end
for c=1:nbin,
    Z(c,:)=abs(sinc((f(c)-partfreq)*alen(c))+.5*sinc((f(c)-partfreq)*alen(c)+1)+.5*sinc((f(c)-partfreq)*alen(c)-1)).^2;
end

%%% Defining harmonic blob spectra and spectral envelopes %%%
totwidth=22;    % maximum total bandwidth of a harmonic spectrum (about 3.4 octaves)
hspace=hwidth/2;
maxblob=round(totwidth/hspace);
bnum=zeros(1,nbpart);    % blob index for each partial
for n=1:nbnotes,
    bnum(ppos(n)+1:ppos(n)+nharm(n))=9.26*(log(.00437*partfreq(ppos(n)+1:ppos(n)+nharm(n))*fs+1)-log(.00437*partfreq(ppos(n)+1)*fs+1))/hspace;
end
nblob=min(maxblob,round(bnum(ppos(2:end)))+1);
bpos=[0,cumsum(nblob)];
hblob=bpos(end);
bfreq=zeros(1,hblob);   % center frequency of each blob
for n=1:nbnotes,
    bfreq(bpos(n)+1:bpos(n)+nblob(n))=((.00437*partfreq(ppos(n)+1)*fs+1)*exp(hspace/9.26*(0:nblob(n)-1))-1)/(.00437*fs);
end
order=8;    % gammatone order (order 4 in magnitude domain)
%order=4;    % gammatone order (order 4 in magnitude domain)
k=sqrt(pi)*gamma(order-.5)/gamma(order);
Vh=zeros(nbin,hblob);   % blob spectra
Bh=zeros(hblob,nbnotes*hcomp);  % spectral envelopes
for n=1:nbnotes,
    weights=zeros(nharm(n),nblob(n));
    for c=1:nblob(n),
        r=(bnum(ppos(n)+1:ppos(n)+nharm(n))-c+1)*hspace/hwidth;
        weights(:,c)=(1+(k*r).^2).^-order;
    end
    Vh(:,bpos(n)+1:bpos(n)+nblob(n))=Z(:,ppos(n)+1:ppos(n)+nharm(n))*weights;
    Bh(bpos(n)+1:bpos(n)+nblob(n),(n-1)*hcomp+1:n*hcomp)=10.^(-eslope/10*log2(bfreq(bpos(n)+1:bpos(n)+nblob(n))/bfreq(bpos(n)+1)).'*(1:hcomp));
end

%%% Defining wideband blob spectra and spectral envelopes %%%
totwidth=e(end);    % maximum total bandwidth of a wideband spectrum
wspace=wwidth/2;
wblob=round(totwidth/wspace);
order=8;    % gammatone order (order 4 in magnitude domain)
k=sqrt(pi)*gamma(order-.5)/gamma(order);
Vw=zeros(nbin,wblob);   % blob spectra
Bw=zeros(wblob,wcomp);  % spectral envelopes
for c=1:wblob,
    r=(e-wspace*(c-1))/wwidth;
    Vw(:,c)=(1+(k*r).^2).^-order;
    Bw(c,:)=rand(1,wcomp);
    % Bw(c,:) =
    % 10.^(-eslope/10*log2((exp(wspace*(c-1)/9.26)-1)/(.00437*fs));   % to
    % change (and add a constant for log2 computation)
end

%%% Grouping harmonic and wideband components %%%
V=[Vh Vw];
B=[Bh zeros(hblob,wcomp); zeros(wblob,nbnotes*hcomp) Bw];

harm_num = size(Bh, 2);

return;