 clear all
clc
close all

Files=dir('binarywaveguide477-furtherplastic/');
N = length(Files);
[t,w]=textread(strcat('binarywaveguide477-furtherplastic/',Files(3).name),'%f%f','headerlines',6);  % read first for size
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
    [T(:,str2double(x(1)) + 1),W(:,str2double(x(1)) + 1)]=textread(strcat('binarywaveguide477-furtherplastic/',name),'%f%f','headerlines',6);
    
    B(:,str2double(x(1)) + 1) = fftshift(fft(W(:,str2double(x(1)) + 1)));
    t = T(:,str2double(x(1)) + 1);
    fmax = 0.5/(t(2)-t(1));
    f(:,str2double(x(1)) + 1) = linspace(-fmax,fmax,size(W,1));
end

objective241 = readmatrix('objective477.csv');
expected241 = readmatrix('expected477_base.csv');
%%
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
plot(theta_pm, nu_pm,'linewidth',2,'color','red')
axis square
% colormap(flipud(brewermap([],'GnBu')))
legend('Data','n=0')
order=-1;
c=3e8;
period = 0.95e-3;
% 
%% plotting
figure
polarplot(deg2rad(4:151),mag2db(abs(B(2111,1:151-3)) / max(abs(B(2111,1:151-3)))),'Color','#f7a400','LineWidth',2)
hold on

polarplot(deg2rad(1:151),mag2db(expected241(1:151,2) / max(expected241(1:151,2))),'Color','#1d91c0','LineWidth',2)

polarplot(deg2rad(1:151),mag2db(objective241(1:151,2) / max(objective241(1:151,2))),'Color','#0c2c84','LineWidth',2)

hold on

ax = gca;
pax.RAxisLocation   =   0;
pax.ThetaDir = 'counterclockwise';
pax.ThetaZeroLocation = 'right';
rlim([-30 0])
thetalim([0 180])
thetaticks(0:30:180)
legend('Experimental','Expected','Objective','location','southwest')
set(gcf,'color','w');
