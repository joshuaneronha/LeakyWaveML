clear all
clc
close all

Files=dir('baselwa');
N = length(Files);
[t,w]=textread(strcat('baselwa/',Files(3).name),'%f%f','headerlines',6);  % read first for size
T=zeros(length(t),N-2); W=T; f=T; B=T;   % preallocate space for time,waveform vectors
T(:,1)=t;
W(:,1)=w;

B(:,1) = fftshift(fft(W(:,1)));
t = T(:,1);
fmax = 0.5/(t(2)-t(1));
f(:,1) = linspace(-fmax,fmax,size(W,1));

for k = 3:N
    name = Files(k).name;
    x = split(name,'.');
    [T(:,str2double(x(1)) + 1),W(:,str2double(x(1)) + 1)]=textread(strcat('baselwa/',name),'%f%f','headerlines',6);
    
    B(:,str2double(x(1)) + 1) = fftshift(fft(W(:,str2double(x(1)) + 1)));
    t = T(:,str2double(x(1)) + 1);
    fmax = 0.5/(t(2)-t(1));
    f(:,str2double(x(1)) + 1) = linspace(-fmax,fmax,size(W,1));
end

plot1 = pcolor(1:90,f(2095:2150),abs(B(2095:2150,1:90)));
set(plot1, 'EdgeColor', 'none');

theta_pm = linspace(0,90,1000); 
order_m = 1; % order of the PPWG mode
b = 1e-3; % slit height
neff = 1; % effective refractive index between the plates
nu_pm = 1e-12*3e8*order_m./(2*b*sqrt(neff^2-cosd(theta_pm).^2));
hold on 
plot(theta_pm, nu_pm,'linewidth',3,'color','#f7a400')
axis square
ylabel('Frequency (THz)')
xlabel('Angle (degrees)')
legend('n=0')
colormap(flipud(brewermap([],'GnBu')))