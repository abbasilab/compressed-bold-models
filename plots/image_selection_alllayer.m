layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};
Wroot='M:/map/vim_2/W_ACC/STR/100/EBA/'
imagenetroot='M:/map/imagenet/Vim_2/100/'
Wr = load([Wroot,'S2_W_all.mat']);
W=Wr.W;
for j=1:50
indx=load('M:/map/vim_2/W_ACC/STR/80/EBA_indx.mat');
vox=indx.ix(j);
img_feat_map=h5read([imagenetroot,'CaffeNet_imagenet_pca_alllayer.h5'],['/data']);
act=(img_feat_map*W(:,vox));
[S,ix]=sort(act);
ix=ix+1000;
  for i=8980:8999
      ix(i);
      a=imread(['M:/data/imagenet/vim2_s1/ILSVRC2012_val_0000',num2str(ix(i)),'.JPEG']);
      imwrite(a,['M:/map/IMAGE/Vim2/100/EBA/',num2str(vox),'__',num2str(i-8979),'.JPEG']);
  end
end