clear all
clc
close all

Files=dir('binarywaveguide194');
N = length(Files);
[t,w]=textread(Files(3).name,'%f%f','headerlines',6);  % read first for size
T=zeros(length(t),N-2); W=T; f=T; B=T;   % preallocate space for time,waveform vectors
T(:,1)=t;
W(:,1)=w;

B(:,1) = fftshift(fft(W(:,1)));
t = T(:,1);
fmax = 0.5/(t(2)-t(1));
f(:,1) = linspace(-fmax,fmax,size(W,1));

for k = 4:N
    name = Files(k).name;
    x = split(name,'.');
    [T(:,str2double(x(1)) + 1),W(:,str2double(x(1)) + 1)]=textread(name,'%f%f','headerlines',6);
    
    B(:,str2double(x(1)) + 1) = fftshift(fft(W(:,str2double(x(1)) + 1)));
    t = T(:,str2double(x(1)) + 1);
    fmax = 0.5/(t(2)-t(1));
    f(:,str2double(x(1)) + 1) = linspace(-fmax,fmax,size(W,1));
end
figure
plot1 = pcolor(1:151,f(2096:2150),abs(B(2096:2150,1:151)));
set(plot1, 'EdgeColor', 'none');
ylabel('Frequency (THz)')
xlabel('Angle (degrees)')

theta_pm = linspace(0,90,1000); 
order_m = 1; % order of the PPWG mode
b = 1e-3; % slit height
neff = 1; % effective refractive index between the plates
nu_pm = 1e-12*3e8*order_m./(2*b*sqrt(neff^2-cosd(theta_pm).^2));
hold on 
plot(theta_pm, nu_pm,'linewidth',2,'color','#F7A400')
axis square
colormap(flipud(brewermap([],'GnBu')))

order=-1;
c=3e8;
period = 0.95e-3;
% 
% theta_pm_back = linspace(0,150,1400)
% nu_vector=(150:0.1:280);
% nu_backfire = -1e-12*((sqrt(2)*sqrt((8*c^2*b^2*order^2)-(c^2.*period.^2.*cosd(2*theta_pm_back))+(c^2.*period.^2))./(b.*period))-((4*c*order.*cosd(theta_pm_back))./period))./(4*((cosd(theta_pm_back).^2)-1));
% plot(theta_pm_back(:,1:1400),nu_backfire(:,1:1400),'linewidth',1.5,'color','white')

objective194 = readmatrix('objective194.csv');
expected194 = readmatrix('expected194.csv');

figure
semilogy(abs(B(2112,:)) / max(abs(B(2112,:))))
hold on
semilogy(objective194(1:180,2) / max(objective194(1:180,2)))

semilogy(expected194(1:180,2) / max(expected194(1:180,2)))