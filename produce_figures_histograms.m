%--------------------------------------------------------------------------
% For Paper
% "On the Natural Gradient of the Evidence Lower Bound"
% by Nihat Ay, Jesse van Oostrum and Adwait Datar
%--------------------------------------------------------------------------
load('data\data_for_cosine_sim_histogram_bad.mat')
% Initialization
cos_sim=zeros(1,samples);
cos_sim_rec=zeros(1,samples);
cos_sim_GAP=zeros(1,samples);
cos_sim_GAP2=zeros(1,samples);

enorm=zeros(1,samples);
enorm_rec=zeros(1,samples);
enorm_GAP=zeros(1,samples);
enorm_GAP2=zeros(1,samples);

% Post process data
for i=1:samples
    % Store for later post-processing
    grad_V=grad_V_all(:,i);
    grad_rec=grad_rec_all(:,i);
    grad=grad_all(:,i);
    G_num=G_num_all(:,:,i);
    G_V_num=G_V_num_all(:,:,i);

    % Compute norms of the gradients on the fisher metric at the visible
    % nodes
    norm_grad_V=sqrt(grad_V'*G_V_num*grad_V);
    norm_grad=sqrt(grad'*G_V_num*grad);
    norm_grad_rec=sqrt(grad_rec'*G_V_num*grad_rec);
    
    % Cosine similarities
    cos_sim(1,i)=(grad'*G_V_num*grad_V)/(norm_grad*norm_grad_V);
    cos_sim_rec(1,i)=(grad_rec'*G_V_num*grad_V)/(norm_grad_rec*norm_grad_V);
    cos_sim_GAP(1,i)=(grad_rec'*G_V_num*grad)/(norm_grad_rec*norm_grad);
    cos_sim_GAP2(1,i)=(grad_rec'*G_num*grad)/((grad_rec'*G_num*grad_rec)*(grad'*G_num*grad));
    
    % Error norms
    enorm_samp=grad_V-grad;
    enorm(1,i)=(enorm_samp'*G_V_num*enorm_samp);
    enorm_rec(1,i)=(grad_rec'*G_V_num*grad_V)/(norm_grad_rec*norm_grad_V);
    enorm_GAP(1,i)=(grad_rec'*G_V_num*grad)/(norm_grad_rec*norm_grad);
    enorm_GAP2(1,i)=(grad_rec'*G_num*grad)/((grad_rec'*G_num*grad_rec)*(grad'*G_num*grad));


    %%
    
end

% Plot results
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