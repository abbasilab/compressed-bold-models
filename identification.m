function identification(weight_path, feature_path, voxles_index, voxel_responses_path, test_length, output_path)

       X_test = h5read(feature_path, '/data');
       W = load(weight_path);
       W = W.W;
       
       Y=load(voxel_responses_path);
       Y_test=Y.rv(:,voxles_index);
        
       X= X_test*W;
       R =[];
       a = find(isnan(X(1,:)));
       X(:,a)=[];
       Y_test(:,a)=[];
        
       for i = 1:test_length
           X_corr = X(i,:);  
           for j = 1:test_length
               Y_corr = Y_test(j,:);
               R(i,j) = corr(X_corr',Y_corr');
            end
       end
       
       save(output_path,'R', '-v7.3');
     end
   

 



