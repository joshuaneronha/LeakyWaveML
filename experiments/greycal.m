clear all
clc
close all

Files=dir('gc5');
N = length(Files);

% blackness = zeros(8,1);
maxes = zeros(8,1);

vals = [0 50 75 100 113 125 140 165];

for k = 1:length(vals)
%     name = Files(k).name;
%     x = split(name,'.');
%     blackness(k-2) = str2double(x(1));
    
    
    [t,w]=textread(strcat('gc5/',num2str(vals(k)),'.picotd'),'%f%f','headerlines',6);
%     maxes(k-2) = max(w.^2);
    
    B = fftshift(fft(w));
    fmax = 0.5/(t(2)-t(1));
    f = linspace(-fmax,fmax,size(w,1));
    maxes(k) = abs(B(2112))^2;
    
end

maxes = maxes / maxes(1);
p = polyfit(vals,maxes,5);
cobf = polyval(p,0:165);
scatter(vals,maxes)
yline(0.5)
hold on
plot(0:165,cobf)
