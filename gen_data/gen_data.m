    % XL-MIMO channel generation
    clear all
    clc
    N=256; % BS antennas
    lamda=0.01;
    Anten_dis=lamda/2; % antenna space
    Anten_dis_norm=0.5; % normalized antenna space
    L_f=10; % maximum number of paths for far-field part
    L_n=10; % maximum number of paths for near-field part
    num_sta=10;
    num_ffading=200;
    num_train=num_sta*num_ffading;
    num_Channel = 10000; %num_Channel % num_train == 0
    Channel_mat_total = zeros(num_train,N);
    
    for j = 1:num_Channel/(num_sta*num_ffading)
    Channel_mat=zeros(num_train,N);
    Channel_near_mat=zeros(num_train,N);
    Channel_far_mat=zeros(num_train,N);
        for i=1:num_sta
            Lf= randi([1,L_f]);
            Ln = randi([1,L_n]);
            Cp_far=1; % channel average power of far part
            Cp_near=1; % channel average power of near part
            % angle_index_far=Allcomb_far(unidrnd(num_comb_far),:);
            % theta_far=angle_grid(angle_index_far); % AoA at BS of far part
            theta_far=pi*unifrnd(-1/2,1/2,1,Lf); % AoA at BS of far part
            % angle_index_near=Allcomb_near(unidrnd(num_comb_near),:);
            % theta_near=angle_grid(angle_index_near); % AoA at BS of near part
            theta_near=pi*unifrnd(-1/2,1/2,1,Ln); % AoA at BS of far part
            r_near=70*rand(1,Ln)+10;   
        
            for n=1:num_ffading
                % generate far-field channel
                A_far=1/sqrt(N)*exp((-1j*2*pi*Anten_dis/lamda*[0:1:N-1]')*sin(theta_far));
                f_far=(normrnd(0,Cp_far/sqrt(2),Lf,1)+1j*normrnd(0,Cp_far/sqrt(2),Lf,1));
                h_far=A_far*f_far;
                % generate near-field channel
                delta=(2*[1:N]'-N-1)/2;
                B_near=zeros(N,Ln);
                for l=1:Ln
                    r_dis=sqrt(r_near(l)^2+delta.^2*Anten_dis^2-2*r_near(l)*delta*Anten_dis*sin(theta_near(l)));
                    B_near(:,l)=1/sqrt(N)*exp(-1j*2*pi/lamda*(r_dis-r_near(l)));
                end
                f_near=(normrnd(0,Cp_near/sqrt(2),Ln,1)+1j*normrnd(0,Cp_near/sqrt(2),Ln,1));
                h_near=B_near*f_near;
                h_hyb=sqrt(N/(Lf+Ln))*(h_far+h_near);        
                Channel_mat((i-1)*num_ffading+n,:)=h_hyb.';
                Channel_far_mat((i-1)*num_ffading+n,:)=sqrt(N/(Lf+Ln))*h_far.';
                Channel_near_mat((i-1)*num_ffading+n,:)=sqrt(N/(Lf+Ln))*h_near.';
            end
        end
        if j == 1
            Channel_mat_total = Channel_mat;
        else
            Channel_mat_total = [Channel_mat_total ; Channel_mat];
        end
    end
    
    
    pathName = append("(output)XL-MIMO\matlab_channel\Channel_f",num2str(L_f),"n",num2str(L_n),"_Total_Model",num2str(num_Channel),"_",num2str(N),"ANTS_",num2str(num_sta),"by",num2str(num_ffading));
        if ~exist('(output)XL-MIMO\matlab_channel\', 'dir')
           mkdir('(output)XL-MIMO\matlab_channel\')
        end
    save([pathName],'Channel_mat_total',"num_Channel")
    winopen("(output)XL-MIMO\matlab_channel\");