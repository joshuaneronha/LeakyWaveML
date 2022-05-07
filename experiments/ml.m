B = fftshift(fft(A(:,2)));
t = A(:,1);
fmax = 0.5/(t(2)-t(1));
f = linspace(-fmax,fmax,size(A,1))'

semilogy(f,abs(B).^2))