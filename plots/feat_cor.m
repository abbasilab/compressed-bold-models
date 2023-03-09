
function feat_cor(feat_path, imagenet_feat_path, top_voxels_path, output_path)

load(feat_path);
load(top_voxels_path)

img_feat_map=h5read(imagenet_feat_path,['/data']);
image_feat=[];

COR=[]

for vox=1:100
    md(vox)
    W=w(:,md(vox));
    act=(img_feat_map*W);
    [S,ix]=sort(act);
    ix=flipud(ix);
    for i=1:10   
      image_feat(i,:)=img_feat_map(ix(i),:);
    end
   cor1=cor_fun(image_feat);
   cor_=mean2(triu(cor1));
   COR(vox,:)=cor_;
end
save(output_path,'COR');