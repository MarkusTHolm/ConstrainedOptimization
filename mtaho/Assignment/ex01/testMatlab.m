clc;clear all

% Load problem: H, g, A, b
load problem.mat

% Assemble system KKT system
[n, m] = size(A);
K = sparse(n+m, n+m);
r = sparse(n+m, 1);

K(1:n, 1:n) = H;
K(n+1:n+m, 1:n) = -A';
K(1:n, n+1:n+m) = -A;

r(1:n, 1) = -g;
r(n+1:n+m, 1) = -b;

% LU-solver:
[L,U,p]=lu(K,'vector');
uLU = U\(L\r(p));

% LDL-solver:
[L,D,p] = ldl(K,'lower','vector');
uLDL(p) = L'\( D \(L\r(p)));
uLDL = uLDL';

% Compute error norm:
err = norm(uLU-uLDL); % err = 9.899e-16

sprintf('err = %1.4e', err)

xLU = uLU(1:n);
xLDL = uLDL(1:n);
