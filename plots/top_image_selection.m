function top_image_selection(top_voxel_index, imagnet_feature_path, output_path, top_voxel_weight_path, image_path, N_image)

load(top_voxel_weight_path);
img_feat_map=h5read(imagnet_feature_path,['/data']);
vox= top_voxel_index
act=(img_feat_map*W);
[S,ix]=sort(act);
for i= N_image-5:N_image
  img=imread([image_path,num2str(ix(i)),'.JPEG']);
  imwrite(img,[output_path,num2str(vox),'_',num2str(i-N_image-5),'.JPEG']);
end

      




