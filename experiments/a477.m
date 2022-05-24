clear all
clc
close all

Files=dir('binarywaveguide477');
N = length(Files);
[t,w]=textread(strcat('binarywaveguide477/',Files(3).name),'%f%f','headerlines',6);  % read first for size
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
    [T(:,str2double(x(1)) + 1),W(:,str2double(x(1)) + 1)]=textread(strcat('binarywaveguide477/',name),'%f%f','headerlines',6);
    
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
legend('Data','n=0')
order=-1;
c=3e8;
period = 0.95e-3;
% 
% theta_pm_back = linspace(0,150,1400)
% nu_vector=(150:0.1:280);
% nu_backfire = -1e-12*((sqrt(2)*sqrt((8*c^2*b^2*order^2)-(c^2.*period.^2.*cosd(2*theta_pm_back))+(c^2.*period.^2))./(b.*period))-((4*c*order.*cosd(theta_pm_back))./period))./(4*((cosd(theta_pm_back).^2)-1));
% plot(theta_pm_back(:,1:1400),nu_backfire(:,1:1400),'linewidth',1.5,'color','white')

objective241 = readmatrix('objective241.csv');
expected241 = readmatrix('expected241.csv');

% figure
% % semilogy(7:151,abs(B(2112,1:end-7)) / max(abs(B(2112,1:end-7))))
% hold on
% % semilogy(objective241(1:151,2) / max(objective241(1:151,2)))
% % semilogy(expected241(1:151,2) / max(expected241(7:151,2)))
% 
% plot(7:151,mag2db(abs(B(2112,1:end-7)) / max(abs(B(2112,1:end-7)))))
% plot(mag2db(objective241(1:151,2) / max(objective241(1:151,2))))
% plot(mag2db(expected241(1:151,2) / max(expected241(7:151,2))))
% 
% legend('Experimental','Objective','Expected')
% 
% % floquet = [5,15,24,28,31,34,48,60,61,63,65,68,70,73,76,80,84,86,88,90,95,99,101,103,106,109,114,117,120,122,126,128,130,132,146];
% % 
% % for i=1:length(floquet)
% %     xline(floquet(i))
% % end
% figure


% 
% Polar_dB(deg2rad(7:151),mag2db(abs(B(2112,1:end-7)) / max(abs(B(2112,1:end-7)))),[-40 0], 5,'k',2)
% hold on
% Polar_dB(deg2rad(7:151),mag2db(objective241(1:151,2) / max(objective241(1:151,2))),[-40 0], 5,'k',2)
figure
polarplot(deg2rad(5:151),mag2db(abs(B(2112,1:end-5)) / max(abs(B(2112,1:end-5)))),'Color','#0c2c84','LineWidth',2)
hold on
polarplot(deg2rad(1:151),mag2db(objective241(1:151,2) / max(objective241(1:151,2))),'Color','#3c7dce','LineWidth',2)
polarplot(deg2rad(1:151),mag2db(expected241(1:151,2) / max(expected241(7:151,2))),'Color','#7fcdbb','LineWidth',2)

pax = gca;
pax.RAxisLocation   =   0;
pax.ThetaDir = 'counterclockwise';
pax.ThetaZeroLocation = 'right';
rlim([-30 0])
thetalim([0 180])
thetaticks(0:30:180)
legend('Experimental','Objective','Simulation','location','southwest')

% 
% figure

% % hold on
% polarplot(deg2rad(1:151),mag2db(objective241(1:151,2) / max(objective241(1:151,2))))
% polarplot(deg2rad(1:151),mag2db(expected241(1:151,2) / max(expected241(7:151,2))))