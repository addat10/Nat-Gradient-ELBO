%--------------------------------------------------------------------------
% For Paper
% "On the Natural Gradient of the Evidence Lower Bound"
% by Nihat Ay, Jesse van Oostrum and Adwait Datar
%--------------------------------------------------------------------------
% This script is used to simulate gradient flows for the cylindrical and
% non-cylindrical model 
clear
close all
clc 
save_data=1; % flag to save data
T=5;delta_t=0.01; % Final time and time-step
%% Define model, inne-product, gradients
model =2; % 1: non-cylindrical, 2: cylindrical
rng(13)
switch model
    case 1 % Define non-cylindrical model with three variables        
        syms x y1 y2 z1 z2
        theta= [x;y1;y2;z1;z2];
        n_param=size(theta,1);
        p=[ theta(1)*theta(2)*theta(4);
            theta(1)*theta(2)*(1-theta(4));
            theta(1)*(1-theta(2))*theta(4);
            theta(1)*(1-theta(2))*(1-theta(4));
            (1-theta(1))*theta(3)*theta(5);
            (1-theta(1))*theta(3)*(1-theta(5));
            (1-theta(1))*(1-theta(3))*theta(5);
            (1-theta(1))*(1-theta(3))*(1-theta(5));
            ];
    case 2 % Define cylindrical model with three variables
        syms x y z        
        theta= [x;y;z];
        n_param=size(theta,1);
        p=[ theta(1)*theta(2)*theta(3);
            theta(1)*theta(2)*(1-theta(3));
            theta(1)*(1-theta(2))*theta(3);
            theta(1)*(1-theta(2))*(1-theta(3));
            (1-theta(1))*theta(2)*theta(3);
            (1-theta(1))*theta(2)*(1-theta(3));
            (1-theta(1))*(1-theta(2))*theta(3);
            (1-theta(1))*(1-theta(2))*(1-theta(3));
            ];
end
% Define the Fisher-Rao inner-product on the model-space
dphi=jacobian(p,theta); % Jacobian of the parameterization/inverse chart
G=simplify(transpose(dphi))*inv(diag(p))*dphi;

% Define the Fisher-Rao inner-product on the projected visible-nodes space
Pi=[eye(4), eye(4)]; % Marginalization map/projection to the visible nodes
p_V=Pi*p; % distribution at the visible nodes
dphi_V=jacobian(p_V,theta); % Jacobian 
G_V=simplify(transpose(dphi_V)*inv(diag(p_V))*dphi_V); 

% Define the target and initial distribution
switch model
    case 1 %non-cylindrical model
            sims=8; % Number of simulations each with a random initial distribution
            % theta_opt=[0.1;0.25;0.5;0.75;0.25];
            theta_opt=round(rand(n_param,1),2); % Reference (optimal) parameters
            theta_0=rand(n_param,sims); % Initial random distribution             
    case 2 %cylindrical model
            sims=1; % Number of simulations each with a random initial distribution
            theta_opt=[0.95;0.1;0.9]; % Reference (optimal) parameters
            theta_0=[0.02;0.95;0.02]; % Initial random distribution
            if sims>1
                theta_0=rand(n_param,sims); % Initial random distribution
            end
            % Other tested configuration
            % theta_opt=[0.5;0.1;0.5]; 
            % theta_0=[0.2200;0.8700;0.2100];
end
p_star=subs(p,theta,theta_opt); % Get the target distribution (Hidden+visible)
p_star_V=Pi*p_star; % target distribution of the visible nodes 

% Compute the Loss and its jacobians on the visible nodes (Reference
% model)
L_V=transpose(p_star_V)*log(p_star_V./p_V);
J_V=transpose(jacobian(L_V,theta)); % Jacobian of the Loss wrt parameters

% Compute the Loss and its jacobians on the hidden+visible nodes model 
L=transpose(p_star)*log(p_star./p);
J=transpose(jacobian(L,theta)); % Jacobian of the Loss wrt parameters

% Compute the Loss and its jacobians for the perfect recognition model
Pi_Q_p=[p_star_V;p_star_V].*(p./[p_V;p_V]);
L_rec=transpose(Pi_Q_p)*log(Pi_Q_p./p);
J_rec=transpose(jacobian(L_rec,theta)); % Jacobian of the Loss wrt parameters

%% Simulate gradient flows
% Gradient flow on the hidden+visible nodes model and evaluate on L_V
[theta_traj,L_num,p_num]=solve_ode(theta_0,T,delta_t,G,J,L_V,p);
% Gradient flow on the perfect recognition nodes model
[theta_traj_rec,L_num_rec,p_num_rec]=solve_ode(theta_0,T,delta_t,G,J_rec,L_V,p);
% Gradient flow on the visible nodes model
[theta_traj_V,L_num_V,p_num_V]=solve_ode(theta_0,T,delta_t,G_V,J_V,L_V,p);
%% Save Data
if save_data
    switch model
        case 1
            save('./data/traj_non_cylindrical_model')
        case 2
            save('./data/traj_cylindrical_model')
    end
end
%% Produce figures from the simulated data
switch model
    case 1 % simulation id to be plotted for the non-cylindrical model
        sim_id=2; 
        % sim_id=8;
    case 2 % simulation id to be plotted for the cylindrical model
        sim_id=1;
end
produce_figures

%% User-defined functions
function [x_sol_all,f1_all,f2_all]=solve_ode(x0,T,delta_t,Hess,grad,fun1,fun2)   
% This function solves the ode with explicit Euler method and returns the
% tranjectories along with the evaluation of fun1 and fun2 along trajs
    
    theta=symvar(grad); % extract the symbolic variables
    iters=T/delta_t; % Number of iterations
    n=size(x0,1); % size of state (here parameter)
    sims=size(x0,2); % Number of simulations
    
    % Initialize all variables to zero
    x_sol_all=zeros(n,iters,sims);
    f1_all=zeros(iters,size(fun2,1),sims);
    f2_all=zeros(iters,size(fun2,1),sims); 
    
    for i=1:sims % Loop over number of simulations
        x_sol=zeros(n,iters); x_sol(:,1)=x0(:,i); % Initialization    
        f1=zeros(iters,size(fun2,1));
        f2=zeros(iters,size(fun2,1));                     
        for k=1:iters-1 % Time-stepping
            Hess_num=double(subs(Hess,theta,x_sol(:,k)')); % Numerical Hess
            grad_num=double(subs(grad,theta,x_sol(:,k)')); % Numerical grad
            x_sol(:,k+1)=x_sol(:,k)-delta_t*pinv(Hess_num)*grad_num;        
            f1(k,1)=double(subs(fun1,theta,x_sol(:,k)'));
            f2(k,:)=double(subs(fun2,theta,x_sol(:,k)'));
        end
        f1(end,1)=double(subs(fun1,theta,x_sol(:,end)'));
        f2(end,:)=double(subs(fun2,theta,x_sol(:,end)'));
        x_sol_all(:,:,i)=x_sol;
        f1_all(:,:,i)=f1;
        f2_all(:,:,i)=f2; 
    end    
end