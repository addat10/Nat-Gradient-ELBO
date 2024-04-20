%--------------------------------------------------------------------------
% For Paper
% "On the Natural Gradient of the Evidence Lower Bound"
% by Nihat Ay, Jesse van Oostrum and Adwait Datar
% Author for the code: Adwait Datar
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
            save('non_cylindrical_model')
        case 2
            save('cylindrical_model')
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