clc;clear

load problem.mat

% x0(abs(x0) < 1e-6) = 0;
% [x,z] = sfsimplex(g,A,b,x0);

[m, n] = size(A)
x0 = ones(n, 1)

[x,info,mu,lambda,iter] = LPippd(g,A,b,x0)

x2 = x(1:n+1)