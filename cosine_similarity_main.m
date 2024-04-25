%--------------------------------------------------------------------------
% For Paper
% "On the Natural Gradient of the Evidence Lower Bound"
% by Nihat Ay, Jesse van Oostrum and Adwait Datar
%--------------------------------------------------------------------------
% This script generates the cosine similarity histograms for the 5 variable
% non-cylindrical Bayesian model
clear all
close all
clc
rng(10)
%% Generate samples on the model manifold and the n-simplex
sample_on_model_and_n_simplex % Generate data-samples wrt Fisher Inf
%% Define the model and the metric
samples=50000; % Number of samples must be less than the data set (50000)
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
% Define the Fisher-Rao inner-product on the hidden+visible nodes model
dphi=jacobian(p,theta); % Jacobian
G=simplify(transpose(dphi))*inv(diag(p))*dphi;

% Define the Fisher-Rao inner-product on the projected visible-nodes space
Pi=[eye(4), eye(4)]; % Marginalization map/projection to the visible nodes
p_V=Pi*p; % distribution at the visible nodes
dphi_V=jacobian(p_V,theta);
G_V=simplify(transpose(dphi_V)*inv(diag(p_V))*dphi_V);

%% Load data
data_ic=load('./data/samples_non_cylindrical');
theta_sample=data_ic.sampleValues';
data_target=load('./data/samples_non_cylindrical_target');
p_target_sample=[data_target.sampleValues_target,1-sum(data_target.sampleValues_target,2)];

% Initialization
cos_sim=zeros(1,samples);
cos_sim_rec=zeros(1,samples);
cos_sim_GAP=zeros(1,samples);
cos_sim_GAP2=zeros(1,samples);

grad_V_all=zeros(n_param,samples);
grad_rec_all=zeros(n_param,samples);
grad_all=zeros(n_param,samples);

G_num_all=zeros(n_param,n_param,samples);
G_V_num_all=zeros(n_param,n_param,samples);

dphi_num_all=zeros(size(p,1),n_param,samples);
dphi_V_num_all=zeros(size(p_V,1),n_param,samples);

GRAD_V_all=zeros(size(p_V,1),samples);
GRAD_D_Q_p_all=zeros(size(p,1),samples);
GRAD_D_q_p_all=zeros(size(p,1),samples);
%% Compute the cosine similarity on the generated samples
for i=1:samples
    % Define the target and initial distribution for the visible nodes
    % p_star=[0.1816;0.0056;0.5750;0.0178;0.1750;0.0054;0.0384;0.0012];%p_target_sample(i,:)';
    p_star=p_target_sample(41577,:)';
    p_star_V=Pi*p_star; % reference distribution of the visible nodes 
    
    % Compute the Loss on the visible nodes and its jacobian
    L_V=transpose(p_star_V)*log(p_star_V./p_V);
    J_V=transpose(jacobian(L_V,theta)); % Jacobian
    
    % Compute the Loss on the hidden+visible nodes and its jacobian
    L=transpose(p_star)*log(p_star./p);
    J=transpose(jacobian(L,theta)); % Jacobian 
    
    % Compute the Loss with the perfect recognition model and its jacobian
    Pi_Q_p=[p_star_V;p_star_V].*(p./[p_V;p_V]);
    L_rec=transpose(Pi_Q_p)*log(Pi_Q_p./p);
    J_rec=transpose(jacobian(L_rec,theta)); % Jacobian 
    
    % Evaluate the FR metric on the samples
    G_num=double(subs(G,theta,theta_sample(:,i))); % Numerical Hess
    G_V_num=double(subs(G_V,theta,theta_sample(:,i))); % Numerical Hess on visible nodes

    % Evaluate the Jacobians on the samples
    J_V_num=double(subs(J_V,theta,theta_sample(:,i))); 
    J_num=double(subs(J,theta,theta_sample(:,i)));    
    J_rec_num=double(subs(J_rec,theta,theta_sample(:,i))); 
    
    % Evaluate the gradients on the samples
    grad_V=pinv(G_V_num)*J_V_num;
    grad=pinv(G_num)*J_num;    
    grad_rec=pinv(G_num)*J_rec_num;

    % Store for later post-processing
    grad_V_all(:,i)=grad_V;
    grad_rec_all(:,i)=grad_rec;
    grad_all(:,i)=grad;
    G_num_all(:,:,i)=G_num;
    G_V_num_all(:,:,i)=G_V_num;
    dphi_num_all(:,:,i)=double(subs(dphi,theta,theta_sample(:,i)));
    dphi_V_num_all(:,:,i)=double(subs(dphi_V,theta,theta_sample(:,i)));
    GRAD_V_all=dphi_V_num_all(:,:,i)*grad_V;
    GRAD_D_Q_p_all=dphi_num_all(:,:,i)*grad_rec;
    GRAD_D_q_p_all=dphi_num_all(:,:,i)*grad;

    % Compute norms of the gradients on the fisher metric at the visible
    % nodes
    norm_grad_V=sqrt(grad_V'*G_V_num*grad_V);
    norm_grad=sqrt(grad'*G_V_num*grad);
    norm_grad_rec=sqrt(grad_rec'*G_V_num*grad_rec);
    
    cos_sim(1,i)=(grad'*G_V_num*grad_V)/(norm_grad*norm_grad_V);
    cos_sim_rec(1,i)=(grad_rec'*G_V_num*grad_V)/(norm_grad_rec*norm_grad_V);
    cos_sim_GAP(1,i)=(grad_rec'*G_V_num*grad)/(norm_grad_rec*norm_grad);
    cos_sim_GAP2(1,i)=(grad_rec'*G_num*grad)/((grad_rec'*G_num*grad_rec)*(grad'*G_num*grad));
    if mod(i,1000)==0
        i/50000
    end
end
save('./data/data_for_cosine_sim_histogram')
%% Plot results
n_bins=15;
figure()
histogram(cos_sim,n_bins,'Normalization','probability','BinWidth',0.1)
xlabel('cosine similarity')
xlim([-1,1])
ylim([0,0.8])
title('ELBO')

figure()
histogram(cos_sim_rec,n_bins,'Normalization','probability','BinWidth',0.1)
xlabel('cosine similarity between gradients')
xlim([-1,1])
ylim([0,0.8])
title('Perfect recognition model') 

figure()
histogram(cos_sim_GAP,n_bins,'Normalization','probability','BinWidth',0.1)
xlabel('cosine similarity between gradients')
xlim([-1,1])
ylim([0,0.8])
title('GAP in V') 

figure()
histogram(cos_sim_GAP2,n_bins,'Normalization','probability','BinWidth',0.1)
xlabel('cosine similarity between gradients')
xlim([-1,1])
ylim([0,0.8])
title('GAP2') 