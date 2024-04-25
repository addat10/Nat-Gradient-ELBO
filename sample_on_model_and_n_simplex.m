%--------------------------------------------------------------------------
% For Paper
% "On the Natural Gradient of the Evidence Lower Bound"
% by Nihat Ay, Jesse van Oostrum and Adwait Datar
%--------------------------------------------------------------------------
% This script is used to generate samples on the model manifold and on the 
% simplex to generate target distributions
clear
clc
rng(1)
close all
clear all 
%% Define the cut-off function on the boundaries of the simplex boundary
tol_pdf=1e-3; % Tolerance to cut-off pdf on the borders of the simplex
cutoff_function=@(theta)prod(heaviside(1-tol_pdf-theta))*prod(heaviside(theta-tol_pdf));
N = 50000; % number of samples
%% Generate samples on the non-cylindrical model example with 5 variables
% The following pdf is set by printing the determinant of G and
% copying it in the function
pdf=get_pdf_5_var_non_cylindrical_model()
pdf_copied=@(theta)(-(theta(1)*(theta(1) - 1))/(theta(2)*theta(3)*theta(4)*theta(5)*(theta(2) - 1)*(theta(3) - 1)*(theta(4) - 1)*(theta(5) - 1)))^(1/2);
pdf = @(theta) pdf_copied(theta)*cutoff_function(theta); % Cutoff on the boundary
theta0=0.5*ones(5,1);
sampleValues = slicesample(theta0,N,"pdf",pdf,"burnin",1000,"thin",5);
figure()
subplot(3,2,1)
histogram(sampleValues(:,2),50,"Normalization","pdf");
title('Histogram for p(y1|x)')
subplot(3,2,2)
histogram(sampleValues(:,3),50,"Normalization","probability");
title('Histogram for p(y2|x)')
subplot(3,2,3)
histogram(sampleValues(:,4),50,"Normalization","probability");
title('Histogram for p(z1|x)')
subplot(3,2,4)
histogram(sampleValues(:,5),50,"Normalization","probability");
title('Histogram for p(z2|x)')
subplot(3,2,5)
histogram(sampleValues(:,1),50,"Normalization","probability");
title('Histogram for p(x)')
save("samples_non_cylindrical.mat","sampleValues");

%% n-simplex in eta coordinates
% The following pdf is set by printing the determinant of G and
% copying it in the function
n=7;
pdf=get_pdf_over_n_simplex(n)
pdf_copied=@(theta)sqrt(1/(prod(theta)*(1-sum(theta))));  
pdf = @(theta) pdf_copied(theta)*cutoff_function(theta);
theta0=(1/(n+1))*ones(n,1);
sampleValues_target = slicesample(theta0,N,"pdf",pdf,"burnin",1000,"thin",5);
figure()
subplot(4,2,1)
histogram(sampleValues_target(:,1),50,"Normalization","probability");
title('Histogram for eta 1')
subplot(4,2,2)
histogram(sampleValues_target(:,2),50,"Normalization","probability");
title('Histogram for eta 2')
subplot(4,2,3)
histogram(sampleValues_target(:,3),50,"Normalization","probability");
title('Histogram for eta 3')
subplot(4,2,4)
histogram(sampleValues_target(:,4),50,"Normalization","probability");
title('Histogram for eta 4')
subplot(4,2,5)
histogram(sampleValues_target(:,5),50,"Normalization","probability");
title('Histogram for eta 5')
subplot(4,2,6)
histogram(sampleValues_target(:,6),50,"Normalization","probability");
title('Histogram for eta 6')
subplot(4,2,7)
histogram(sampleValues_target(:,7),50,"Normalization","probability");
title('Histogram for eta 7')
save("samples_non_cylindrical_target.mat","sampleValues_target"); 

%% User-defined functions
function pdf=get_pdf_5_var_non_cylindrical_model()
% This function defines the 5 variable noncylindrical model and returns the
% pdf for sampling points on it
syms x y1 y2 z1 z2
theta= [x;y1;y2;z1;z2];
p=[ theta(1)*theta(2)*theta(4);
	theta(1)*theta(2)*(1-theta(4));
	theta(1)*(1-theta(2))*theta(4);
	theta(1)*(1-theta(2))*(1-theta(4));
	(1-theta(1))*theta(3)*theta(5);
	(1-theta(1))*theta(3)*(1-theta(5));
	(1-theta(1))*(1-theta(3))*theta(5);
	(1-theta(1))*(1-theta(3))*(1-theta(5));
	];
dphi=jacobian(p,theta); % Jacobian of the parameterization/inverse chart
G=simplify(transpose(dphi))*inv(diag(p))*dphi;  
pdf=sqrt(det(G));
end
function pdf=get_pdf_over_n_simplex(n)
% This function returns the pdf for sampling points on the n-simplex
theta = sym('theta',[n 1]);           
p=[theta;1-sum(theta)];
dphi=jacobian(p,theta); % Jacobian of the parameterization/inverse chart
G=simplify(transpose(dphi))*inv(diag(p))*dphi; 
pdf=sqrt(det(G));
end