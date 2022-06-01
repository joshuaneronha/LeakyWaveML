r = 0.12;
x = 0.05;
theta = linspace(0,150*(pi/180),151);
% theta = pi/4;

solvedthetap = zeros(1,length(theta));

for i=2:length(theta)
    i

    syms thetap
    
    c = sqrt(2*r^2-2*r^2*cos(theta(i)));
    phi = asin((r/c)*sin(theta(i)));
    answer = solve(sin(thetap)/c == sin(pi - phi - thetap)/(r+x),thetap);
    if length(answer) == 1
        solvedthetap(i) = answer;
    elseif double(answer(1)) > double(answer(2))
        solvedthetap(i) = answer(1);
    else
        solvedthetap(i) = answer(2);
    end
    
end