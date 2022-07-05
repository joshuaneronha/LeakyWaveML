clc;clear all
freq = 200e9; %frequency 
lambda = 3e8/freq; %wavelength
k0 = 2*pi/lambda; % free-space wavevector
h = 1e-3; % plate separation

% assuming total length is 18 mm and pixel length of .5 mm (if you change this, change also here)
Lambda = 1e-3:1e-3:9e-3; 
% this goes from minimum period 10101010, which is a period of 1mm
% to the maximum period, which is half of the slot is open 11111...0000
p = -80:1:80; %floquet mode number, we just try a lot of floquet modes until it exists.

k=1;
for i = 1:length(Lambda) 
    for j = 1:length(p)
        betaz(i,j) = (sqrt(k0^2-(pi/h)^2)+2*pi*p(j)/Lambda(i)); % calculate the dispersion constant for every possible Lambda,p combination
        neff(i,j) = betaz(i,j)/k0; % calculate the corresponding effective index
        if neff(i,j) > -1 && neff(i,j) < 1 % The mode exists (and leaks) only when it is between 0 and 1
            neff_works(k,:) = [neff(i,j) acosd(neff(i,j)) Lambda(i) p(j)];% This is the matrix of solution. 
            % 1st column is the value of neff that works
            % 2nd column is the peak angle (assuming theta=0 is parallel to the slot)
            % corresponding Lambda 
            % corresponding p
            k=k+1;
        end
    end
end
size(neff_works,1) % this number is the total number of possible angles

plot(neff_works(:,2),'.') % this plots the possible angles

%these are the floquet peaks
sort(unique(floor(neff_works(:,2))))
