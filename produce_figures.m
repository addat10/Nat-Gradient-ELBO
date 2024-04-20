%--------------------------------------------------------------------------
% For Paper
% "On the Natural Gradient of the Evidence Lower Bound"
% by Nihat Ay, Jesse van Oostrum and Adwait Datar
% Author for the code: Adwait Datar
%--------------------------------------------------------------------------
% This script is used to plot the simulated trajectories
clc
close all
%% Load data if not already present in workspace
if ~exist('model')
    model=2; % 1: non-cylindrical model, 2: cylindrical
    switch model
        case 1
            load('non_cylindrical_model.mat')
            % sim_id=2; 
            sim_id=8;
        case 2
            load('cylindrical_model.mat')
            sim_id=1;
    end 
end
%% Plot trajectories:
subsamp=10;
% Plot the parameter trajectories of the two gradient fields
figure(1)
plot(theta_traj(:,1:subsamp:end,sim_id)','r','DisplayName','NGRAD')
set(gca,'ColorOrderIndex',1)
hold on
plot(theta_traj_rec(:,1:subsamp:end,sim_id)','g','DisplayName','NGRAD')
plot(theta_traj_V(:,1:subsamp:end,sim_id)','b--','DisplayName','ELBO')
% legend()
xlim([0 40])
xlabel('time')
ylabel('ksi')
title('Parameter trajectories')

% Plot the Loss trajectories with the two gradient fields
figure(2)
plot(L_num(1:subsamp:end,1,sim_id),'r','DisplayName','ELBO')
set(gca,'ColorOrderIndex',1)
hold on
plot(L_num_V(1:subsamp:end,1,sim_id),'b--','DisplayName','NGRAD')
plot(L_num_rec(1:subsamp:end,1,sim_id),'g','DisplayName','2')
% legend()
xlim([0 40])
xlabel('time')
ylabel('L')
title('KL divergence trajectory')

% Plot the probability trajectories with the two gradient fields on visible
% nodes
p_traj_V=Pi*p_num(1:subsamp:end,:,sim_id)'; % Projection on the visible nodes
p_traj_V_rec=Pi*p_num_rec(1:subsamp:end,:,sim_id)'; % Projection on the visible nodes
p_traj_V_V=Pi*p_num_V(1:subsamp:end,:,sim_id)'; % Projection on the visible nodes
figure(3)
plot(p_traj_V','r','DisplayName','NGRAD')
set(gca,'ColorOrderIndex',1)
hold on
plot(p_traj_V_rec','g','DisplayName','2')
plot(p_traj_V_V','b--','DisplayName','ELBO')
% legend()
xlim([0 40])
xlabel('time')
ylabel('p_V')
title('Distribution at the visible nodes')

% Project the distribution orthogonally (Euclidean) to get a 3-simplex 
% Obtain vectors orthogonal to ones(4,1) via SVD and project on them
[U,S,V]=svd(ones(4,1));
corners=U(:,2:end)';

% Compute translation and rotation parameters
origin=corners(:,1);
new_x=corners(:,2)-origin;
new_z=null([(corners(:,2)-origin)';(corners(:,3)-origin)']);
new_y=null([new_x';new_z']);
W=[new_x';new_y';new_z'];

% Translate and Rotate corners and optimal point:
p_star_proj=U(:,2:end)'*p_star_V;
[p_star_proj] =rotate_translate(p_star_proj,W,origin);
[corners] =rotate_translate(corners,W,origin);

% First draw the lines of the 3-simplex
figure(4)
% title('Distribution trajectories at the visible nodes')

% Plot the model space if required
switch model
    case 1
        plot_full_model(W,origin)
        hold on
    case 2
        plot_independece_model(W,origin)
        hold on
end

% For the final plots in the paper, use i=2 and i=8 from the data non_cylindrical_model_30.mat
% For the final plots in the paper, use i=1:sims from the data cylindrical_model_1.mat
for i=[sim_id]    
    p_traj_V_proj=U(:,2:end)'*Pi*p_num(1:subsamp:end,:,i)';
    p_traj_V_proj_rec=U(:,2:end)'*Pi*p_num_rec(1:subsamp:end,:,i)';
    p_traj_V_V_proj=U(:,2:end)'*Pi*p_num_V(1:subsamp:end,:,i)';
    % Translate and Rotate data
    [p_traj_V_proj] =rotate_translate(p_traj_V_proj,W,origin);
    [p_traj_V_proj_rec] =rotate_translate(p_traj_V_proj_rec,W,origin);
    [p_traj_V_V_proj] =rotate_translate(p_traj_V_V_proj,W,origin);
    plot3(p_traj_V_V_proj(1,:),p_traj_V_V_proj(2,:),p_traj_V_V_proj(3,:),'b--','LineWidth',0.5)    
    hold on
    plot3(p_traj_V_proj(1,:),p_traj_V_proj(2,:),p_traj_V_proj(3,:),'r','LineWidth',0.5)  
    plot3(p_traj_V_proj_rec(1,:),p_traj_V_proj_rec(2,:),p_traj_V_proj_rec(3,:),'g','LineWidth',0.5)
    % plot3(p_traj_V_V_proj(1,1),p_traj_V_V_proj(2,1),p_traj_V_V_proj(3,1),'b.')
    % plot3(p_traj_V_proj(1,1),p_traj_V_proj(2,1),p_traj_V_proj(3,1),'r.')
    plot3(p_star_proj(1,1),p_star_proj(2,1),p_star_proj(3,1),'kX')
end
linewidth=0.5;
linecolor=[0,0.447,0.741];
draw_lines(corners(:,1),corners(:,2),linecolor,linewidth)
hold on
draw_lines(corners(:,1),corners(:,3),linecolor,linewidth)
draw_lines(corners(:,1),corners(:,4),linecolor,linewidth)
draw_lines(corners(:,2),corners(:,3),linecolor,linewidth)
draw_lines(corners(:,2),corners(:,4),linecolor,linewidth)
draw_lines(corners(:,3),corners(:,4),linecolor,linewidth)
% legend('NGRAD','ELBO','Location','northwest')

view([202 20])
zlim([-0.2,1.2])
ylim([0,1.2247])
xlim([-0.75 2])
axis off
%% User-defined functions
function [] = draw_lines(point1,point2,linecolor,linewidth)
        xyz=[point1';point2'];
        line(xyz(:,1),xyz(:,2),xyz(:,3),'Color',linecolor,'Linewidth',linewidth)
end
function []=plot_independece_model(W,origin)
    [U,~,~]=svd(ones(4,1)); % Projection on the kernel of the 1 vector
    % This function plots the independence model as a manifold embedded
    % inside the 3-Simplex
    p=0:0.02:1; % parameter 1
    q=0:0.02:1; % parameter 2
    p_vec=[];
    for i=1:length(p)
        for j=1:length(q)
            p_vec=[p_vec,[p(i)*q(j);p(i)*(1-q(j));(1-p(i))*q(j);(1-p(i))*(1-q(j))]]; % Collect points on the independence model
        end
    end
    p_proj=U(:,2:end)'*p_vec;
    [p_proj] =rotate_translate(p_proj,W,origin);
    n_p=size(p,2);
    linecolor=[0.3010 0.7450 0.9330];
    linewidth=0.1;
    for i=1:n_p        
            draw_lines(p_proj(:,1+n_p*(i-1)),p_proj(:,n_p*i),linecolor,linewidth)
    end
    n_q=size(q,2);
    for j=1:n_q
        draw_lines(p_proj(:,j),p_proj(:,j+n_q*(n_p-1)),linecolor,linewidth)
    end
    view([202 16])
    % view([200 15])
    % xlabel('x')
    % ylabel('y')
    % zlabel('z')
end
function []=plot_full_model(W,origin)    
    [U,~,~]=svd(ones(4,1)); % Projection on the kernel of the 1 vector
    draw_plane_simplex(W,origin,U,1)
    draw_plane_simplex(W,origin,U,2)
    draw_plane_simplex(W,origin,U,3)
    draw_plane_simplex(W,origin,U,4)
    draw_plane_simplex(W,origin,U,5)
    draw_plane_simplex(W,origin,U,6)
    view([202 16])
    % view([200 15])
    % xlabel('x')
    % ylabel('y')
    % zlabel('z')
end
function [points] =rotate_translate(points,W,origin)
    % This function translates points to origin and then rotates them with
    % rotation matrix W;
    points=points-origin;
    points=W*points;
end
function []=draw_plane_simplex(W,origin,U,dimensions)
    x=0:0.05:1; % parameter 1
    linewidth=0.1;
    linecolor=[0.3010 0.7450 0.9330];%
    % linecolor=[0,0.447,0.741];
    for i=1:length(x)
        y=0:0.1:1-x(i); % parameter 2
        for j=1:length(y)
            % z=0:0.1:1-x(i)-y(j); % parameter 3            
            switch dimensions
                case 1 % vary dimensions 1-2
                    p1=[1-x(i)-y(j);0;y(j);x(i)];
                    p_end=[ 0;1-x(i)-y(j);y(j);x(i)];
                case 2 % vary dimensions 3-4
                    p1=[x(i);y(j);0;1-x(i)-y(j)];
                    p_end=[ x(i);y(j);1-x(i)-y(j);0];
                case 3 % vary dimensions 1-3
                    p1=[1-x(i)-y(j);y(j);0;x(i)];
                    p_end=[0;y(j);1-x(i)-y(j);x(i)];
                case 4 % vary dimensions 2-3
                    p1=[x(i);1-x(i)-y(j);0;y(j)];
                    p_end=[x(i);0;1-x(i)-y(j);y(j)];
                case 5 % vary dimensions 1-4
                    p1=[0;x(i);y(j);1-x(i)-y(j)];
                    p_end=[1-x(i)-y(j);x(i);y(j);0];
                case 6 % vary dimensions 2-4
                    p1=[x(i);0;y(j);1-x(i)-y(j)];
                    p_end=[x(i);1-x(i)-y(j);y(j);0];
            end
            p_proj_line=U(:,2:end)'*[p1,p_end];
            [p_proj_line] =rotate_translate(p_proj_line,W,origin);
            if x(i)+y(j)<1
                draw_lines(p_proj_line(:,1),p_proj_line(:,2),linecolor,linewidth)
                hold on
            end
        end
    end
end