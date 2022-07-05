close all; clear; clc;
Ns          =   6e2;
theta2      =   linspace(-pi,pi,Ns);
data2     	=   6*cos(theta2).*cos(3*theta2).*cos(4*theta2);
data2       =   10*log10(abs(data2));
figure
Polar_dB(theta2,data2,[-40 10],5,'k',1)
title('Polar plot in dB','FontName','times new roman')
theta1      =   linspace(-pi,0,Ns);
data1       =   2*cos(theta1).*cos(4*theta1).*cos(10*theta1);
data1     	=   10*log10(abs(data1));
theta2      =   linspace(0,pi,Ns);
data2     	=   6*cos(theta2).*cos(3*theta2).*cos(4*theta2);
data2       =   10*log10(abs(data2));
figure
Polar_dB(theta1,data1,[-40 8],5,'b',2)
hold on
Polar_dB(theta2,data2,[-40 8],5,'-.r',2)
hold off
title('Polar plot in dB','FontName','times new roman')
