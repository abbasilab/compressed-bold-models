function al_preprocessing(dataroot, output_path, srate, start_, stop_, downsampling_step)
% dataroot: path to pca reduced concatenated signal (before down sampled
% and convolved with hrf)
% output: where to save processed features
% strate: video samplig rate
% start_: starting fmri timestep
% stop_: stop fmri timestep
% downsampling_step: video required downsampling to match fmri

layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};
conv1=[h5read([dataroot,'CaffeNet_feature_maps_pca_concatenated.h5'], [layername{1},'/data'])];
conv2=[h5read([dataroot,'CaffeNet_feature_maps_pca_concatenated.h5'], [layername{2},'/data'])];
conv3=[h5read([dataroot,'CaffeNet_feature_maps_pca_concatenated.h5'], [layername{3},'/data'])];
conv4=[h5read([dataroot,'CaffeNet_feature_maps_pca_concatenated.h5'], [layername{4},'/data'])];
conv5=[h5read([dataroot,'CaffeNet_feature_maps_pca_concatenated.h5'], [layername{5},'/data'])];
fc6=[h5read([dataroot,'CaffeNet_feature_maps_pca_concatenated.h5'], [layername{6},'/data'])];
fc7=[h5read([dataroot,'CaffeNet_feature_maps_pca_concatenated.h5'], [layername{7},'/data'])];
fc8=[h5read([dataroot,'CaffeNet_feature_maps_pca_concatenated.h5'], [layername{8},'/data'])];
lay_feat_cont=[conv1,conv2,conv3,conv4,conv5,fc6,fc7,fc8]';

lay_feat_mean=mean(lay_feat_cont,1);
lay_feat_std=std(lay_feat_cont,0,1);

% standardize the time series for each unit  
lay_feat_cont = bsxfun(@minus, lay_feat_cont, lay_feat_mean);
lay_feat_cont = bsxfun(@rdivide, lay_feat_cont, lay_feat_std);
lay_feat_cont(isnan(lay_feat_cont)) = 0; % assign 0 to nan values
if size(lay_feat_cont,1) > size(lay_feat_cont,2)
    R = lay_feat_cont'*lay_feat_cont/size(lay_feat_cont,1);
    [U,S] = svd(R);
    s = diag(S);   
    % keep 99% variance
    ratio = cumsum(s)/sum(s); 
    Nc = find(ratio>0.99,true,'first'); % number of components   
    S_2 = diag(1./sqrt(s(1:Nc))); % S.^(-1/2)
    B = lay_feat_cont*(U(:,1:Nc)*S_2/sqrt(size(lay_feat_cont,1)));
        
 else
      R = lay_feat_cont*lay_feat_cont';
      [U,S] = svd(R);
      s = diag(S);
      % keep 99% variance
      ratio = cumsum(s)/sum(s); 
      Nc = find(ratio>0.99,true,'first'); % number of components
      B = U(:,1:Nc);
    end

% %     save principal components
    save([dataroot,'CaffeNet_feature_maps_pca_all_layer.mat'], 'B', 's', '-v7.3');

% % % % % % % % %  using pre-defined hemodynamic response function% % % % % % % % % %
% % % % % % % % % Dimension reduction for CNN features % % % % % % % % % %

   load([dataroot,'CaffeNet_feature_maps_pca_all_layer.mat'], 'B');     
% %     % Dimension reduction for training data
    
    dim = size(lay_feat_cont);
    lay_feat_cont = reshape(lay_feat_cont, prod(dim(1:end-1)),dim(end));
    Y = lay_feat_cont'*B/sqrt(size(B,1)); % Y: #time-by-#components
    
    ts = conv2(hrf,Y); % convolude with hrf
    ts = ts(start_:stop_,:);
    ts = ts(srate+1:downsampling_step:end,:); % downsampling to match fMRI
            
    h5create([output_path,'feature_maps_pcareduced_all.h5'],['/data'],...
                    [size(Y)],'Datatype','single');
    h5write([output_path,'feature_maps_pcareduced_all.h5'], ['/data'], ts);
   

