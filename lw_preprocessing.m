function lw_preprocessing(dataroot, output_path, N_train_seg, N_test_seg, srate, start_, stop_, downsampling_step)
% dataroot: input_path
% output: where to save processed features
% N_train_seg: number of training segmnets
% N_test_seg: number of test segments
% strate: video samplig rate
% start_: starting fmri timestep
% stop_: stop fmri timestep
% downsampling_step: video required downsampling to match fmri

layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};

% calculate the temporal mean 
for lay = 1 : length(layername)
    N = 0;
    for seg = 1 : N_train_seg
        disp(['Layer: ',layername{lay},'; Seg: ', num2str(seg)]);
        secpath = [dataroot,'seg_', num2str(seg),'.h5'];
        if exist(secpath,'file')
            lay_feat = h5read(secpath,[layername{lay},'/data']);
            dim = size(lay_feat); % convolutional layers: #kernel*fmsize1*fmsize2*#frames
            if seg == 1 
                lay_feat_mean = zeros([dim(1:end-1),1]);
            end
            lay_feat_mean = lay_feat_mean + sum(lay_feat,length(dim));
            N = N  + dim(end);
        end
    end
    lay_feat_mean = lay_feat_mean/N;
     
    h5create([output_path,'feature_maps_avg.h5'],[layername{lay},'/data'],...
        [size(lay_feat_mean)],'Datatype','single');
    h5write([output_path,'feature_maps_avg.h5'], [layername{lay},'/data'], lay_feat_mean);
end    
    
% calculate the temporal standard deviation
for lay = 1 : length(layername)
    lay_feat_mean = h5read([dataroot,'feature_maps_avg.h5'], [layername{lay},'/data']);
    N = 0;
    for seg = 1 : N_train_seg
        disp(['Layer: ',layername{lay},'; Seg: ', num2str(seg)]);
        secpath = [dataroot,'seg', num2str(seg),'.h5'];
        if exist(secpath,'file')
            lay_feat = h5read(secpath,[layername{lay},'/data']);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = lay_feat.^2;
            dim = size(lay_feat);
            if seg == 1 
                lay_feat_std = zeros([dim(1:end-1),1]);
            end
            lay_feat_std = lay_feat_std + sum(lay_feat,length(dim));
            N = N  + dim(end);
        end
    end
    lay_feat_std = sqrt(lay_feat_std/(N-1));
    lay_feat_std(lay_feat_std==0) = 1;
    h5create([output_path,'feature_maps_std.h5'],[layername{lay},'/data'],...
        [size(lay_feat_mean)],'Datatype','single');
    h5write([output_path,'feature_maps_std.h5'], [layername{lay},'/data'], lay_feat_std);
end  

%calculate PCA
for lay = 1 : length(layername)
    lay_feat_mean = h5read([output_path,'feature_maps_avg.h5'], [layername{lay},'/data']);
    lay_feat_std = h5read([output_path,'feature_maps_std.h5'], [layername{lay},'/data']);
    lay_feat_mean = lay_feat_mean(:);
    lay_feat_std = lay_feat_std(:);
    
    % Concatenating the feature maps across training movie segments.  
    for seg = 1 : N_train_seg
        disp(['Layer: ',layername{lay},'; Seg: ', num2str(seg)]);
        secpath = [dataroot,'seg', num2str(seg),'.h5'];
            
        lay_feat = h5read(secpath,[layername{lay},'/data']);
        dim = size(lay_feat);
        Nu = prod(dim(1:end-1)); % number of units
        Nf = dim(end); % number of frames
        lay_feat = reshape(lay_feat,Nu,Nf);% Nu*Nf
        
        
        if seg == 1
            lay_feat_cont = zeros(Nu, Nf*N_train_seg,'single');
        end
        lay_feat_cont(:, (seg-1)*Nf+1:seg*Nf) = lay_feat;
    end
    
    % standardize the time series for each unit  
    lay_feat_cont = bsxfun(@minus, lay_feat_cont, lay_feat_mean);
    lay_feat_cont = bsxfun(@rdivide, lay_feat_cont, lay_feat_std);
    lay_feat_cont(isnan(lay_feat_cont)) = 0; % assign 0 to nan values
    
    %[B, S] = svd(lay_feat_cont,0);
    if size(lay_feat_cont,1) > size(lay_feat_cont,2)
        R = lay_feat_cont'*lay_feat_cont/size(lay_feat_cont,1);
        [U,S] = svd(R);
        s = diag(S);
        
        % keep 99% variance
        ratio = cumsum(s)/sum(s); 
        Nc = find(ratio>0.99,true,'first'); % number of components
        
        S_2 = diag(1./sqrt(s(1:Nc))); % S.^(-1/2)
        B = lay_feat_cont*(U(:,1:Nc)*S_2/sqrt(size(lay_feat_cont,1)));
        % I = B'*B; % check if I is an indentity matrix
        
    else
        R = lay_feat_cont*lay_feat_cont';
        [U,S] = svd(R);
        s = diag(S);
        
        % keep 99% variance
        ratio = cumsum(s)/sum(s); 
        Nc = find(ratio>0.99,true,'first'); % number of components
        
        B = U(:,1:Nc);
    end

    % save principal components
    save([output_path,'feature_maps_pca_layer',num2str(lay),'.mat'], 'B', 's', '-v7.3');

end

% (HRF) with positive peak at 5s.
p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);

for lay = 1 : length(layername)
    lay_feat_mean = h5read([ouput_path,'feature_maps_avg.h5'], [layername{lay},'/data']);
    lay_feat_std = h5read([ouput_path,'feature_maps_std.h5'], [layername{lay},'/data']);
    load([ouput_path,'feature_maps_pca_layer',num2str(lay),'.mat'], 'B');
    
    % Dimension reduction for testing data
    for seg = 1 : N_train_seg
        disp(['Layer: ', num2str(lay),'; Seg: ',num2str(seg)]);
        secpath = [dataroot,'feature_maps_seg', num2str(seg),'.h5'];
        if exist(secpath,'file')==2
            lay_feat = h5read(secpath,[layername{lay},'/data']);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
            
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
            Y = lay_feat'*B/sqrt(size(B,1)); % Y: #time-by-#components
            
            ts = conv2(hrf,Y); % convolude with hrf
            ts = ts(start_:stop_,:);
            ts = ts(srate+1:downsampling_step:end,:); % downsampling to match fMRI
            
            h5create([ouput_path,'feature_maps_pcareduced_seg', num2str(seg),'.h5'],[layername{lay},'/data'],...
                [size(ts)],'Datatype','single');
            h5write([ouput_path,'feature_maps_pcareduced_seg', num2str(seg),'.h5'], [layername{lay},'/data'], ts);
        end
    end   
    
    % Dimension reduction for testing data
    for seg = 1 : N_test_seg
        disp(['Layer: ', num2str(lay),'; Test: ',num2str(seg)]);
        secpath = [dataroot,'seg_', num2str(seg),'.h5'];
        if exist(secpath,'file')==2
            lay_feat = h5read(secpath,[layername{lay},'/data']);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
            
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
            Y = lay_feat'*B/sqrt(size(B,1)); % Y: #time-by-#components
            
            ts = conv2(hrf,Y); % convolude with hrf
            ts = ts(4*srate+1:4*srate+size(Y,1),:);
            ts = ts(srate+1:downsampling_step:end,:); % downsampling
          
            h5create([ouput_path,'feature_maps_pcareduced_test', num2str(seg),'.h5'],[layername{lay},'/data'],...
                [size(ts)],'Datatype','single');
            h5write([ouput_path,'feature_maps_pcareduced_test', num2str(seg),'.h5'], [layername{lay},'/data'], ts);
        end
    end  
    
end

% Concatenate the dimension-reduced CNN features of training movies
for lay = 1 : length(layername)
    for seg = 1 : N_train_seg
        disp(['Layer: ', num2str(lay),'; Seg: ',num2str(seg)]);
        secpath = [output_path,'feature_maps_pcareduced_seg', num2str(seg),'.h5'];      
        lay_feat = h5read(secpath,[layername{lay},'/data']);% #time-by-#components
        dim = size(lay_feat);
        Nf = dim(1); % number of frames
        if seg == 1
           Y = zeros([Nf*N_train_seg, dim(2)],'single'); 
        end
        Y((seg-1)*Nf+1:seg*Nf, :) = lay_feat;
    end
    h5create([output_path,'feature_maps_pcareduced_concatenated.h5'],[layername{lay},'/data'],...
        [size(Y)],'Datatype','single');
    h5write([output_path,'feature_maps_pcareduced_concatenated.h5'], [layername{lay},'/data'], Y);
end