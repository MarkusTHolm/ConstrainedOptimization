function [] = PlotSolutionQP(U)
%--------------------------------------------------------------------------
%   Author:
%       Nicola Cantisani (nicca@dtu.dk)
%
%--------------------------------------------------------------------------
%   Description:
%       Reshapes and plots the open loop solution (inputs) of the optimal 
%       control problem for the 4-tank system (02612 assignment 2024).
%       The outputs of the system (level of the water in tank 1 and 2) 
%       are also plotted.
%
%   Inputs:
%       U       :     solution (column) vector (stacked inputs)
%
%--------------------------------------------------------------------------

%% Plot settings
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(0,'defaultAxesFontSize',15);

%% Load variables
load('plot_output.mat')
load('solCVX.mat')

%% Simulate system and convert to physical variables
u = reshape(U,[2,100]);

x = x_kk;
z = C*x_kk;
for i = 1:N-1
    x(:,i+1) = A*x(:,i) + B*u(:,i);
    z(:,i+1) = C*x(:,i+1);
end

% Convert to physical variables
%x = x + [xs; ds];
z = z + zs;
u = u + us;
r = reshape(R_k,[2 100]) + zs;

%% Plot inputs
f1 = figure();
tiledlayout(2,1)
nexttile
for i=1:2
    stairs(T,u(i,:),'Linewidth',2)
    hold on
end
title('Inputs')
xlabel('Time [min]')
ylabel('Flow rate [cm$^3$/s]')
yline(400)
yline(0)
ylim([-10,410])
xlim([T(1) T(end)])
legend('$F_1$','$F_2$','$u_{min}$','$u_{max}$','Location','southwest')

%% Plot system outputs

nexttile
hold on
for i = 1:2
    stairs(T,z(i,:),'Linewidth',2)
    stairs(T,r(i,:),'Linewidth',2,'LineStyle','--')
end
xlabel('Time [min]')
ylabel('Height [cm]')
title('Outputs')
xlim([T(1) T(end)])
legend('$z_1$','$r_1$','$z_2$','$r_2$')
saveas(f1, 'PlotSolutionQP.eps', 'epsc')


end