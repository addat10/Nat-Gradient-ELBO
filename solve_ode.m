%--------------------------------------------------------------------------
% For Paper
% "On the Natural Gradient of the Evidence Lower Bound"
% by Nihat Ay, Jesse van Oostrum and Adwait Datar
% Author for the code: Adwait Datar
%--------------------------------------------------------------------------
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