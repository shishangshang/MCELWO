function [W,M,Test_Result_error,Result,Train_Result_error]=EWMC_classification(X_train,Y_train,X_test,Y_test,each_class_num,list_train,list_test,num_class,num_view,maxIter)
%Title:When Multi-view Classification Meets Ensemble Learning

%% 5折交叉验证法选择训练集和测试集  
    subtrain_num=list_train;
    sub_each_class_num=floor(subtrain_num/num_class);
    ix=sub_each_class_num;
    % 防止出现某一类为空
    indices=crossvalind('Kfold',ix,5);
     
   for val=1:5
    test_vali = (indices == val);
    train_vali = ~test_vali; 
    idx_test_vali = find(test_vali);
    idx_train_vali = find(train_vali); 
     for v=1:num_view
       X_train_vali{val}{1,v} = []; 
       X_test_vali{val}{1,v} = []; 
       Y_train_vali{val} = []; 
       Y_test_vali{val} = []; 
           for i = 1:num_class
           X_train_vali{val}{1,v}= [X_train_vali{val}{1,v}' X_train{v}(sub_each_class_num*(i-1)+idx_train_vali,:)']';
           X_test_vali{val}{1,v} = [X_test_vali{val}{1,v}' X_train{v}(sub_each_class_num*(i-1)+idx_test_vali,:)']';
           Y_train_vali{val} = [Y_train_vali{val}' Y_train(sub_each_class_num*(i-1)+idx_train_vali,:)']';
           Y_test_vali{val} = [Y_test_vali{val}' Y_train(sub_each_class_num*(i-1)+idx_test_vali,:)']';
           end
     end

%% 矩阵转换
   if length(X_train_vali)>1
      for v=1:num_view
         X_train_vali_v{val}{v,1}=X_train_vali{val}{1,v};
         X_test_vali_v{val}{v,1}=X_test_vali{val}{1,v};
      end
  else     
     X_train_vali_v=X_train_vali;
     X_test_vali_v=X_test_vali;
   end
  
%% (Multi-class SVM）学习初级分类器（方法一）
  for v=1:num_view
  model{val}{v}= libsvmtrain(Y_train_vali{val}, X_train_vali_v{val}{v}, '-s 0 -t 0 -c 1');
  [results_M{1,val}{v}, accuracy, dec_values] = libsvmpredict( Y_test_vali{val}, X_test_vali_v{val}{v}, model{val}{v});
  end
  
%% (KNN)学习初级分类器 （方法二）
% kn=5;
% results_M{1,val} = knn_classify(X_train_vali_v{val},X_test_vali_v{val},kn,Y_train_vali{val}',num_view);

%% (决策树)学习初级分类器(方法三)
%      for v=1:num_view
%      tree{val}{v} = fitctree(full(X_train_vali_v{val}{v}),Y_train_vali{val},'Prune','off');
%      results_M{1,val}{v} = predict(tree{val}{v},full(X_test_vali_v{val}{v}));
%      end
  

%% 矩阵转换
 for v=1:num_view
     strain_number=length(results_M{val}{v});
      X_train_feature{val}{v}=zeros(strain_number,num_class);
      for i=1:strain_number   
      X_train_feature{val}{v}(i,results_M{val}{v}(i))=1;
      end  
 end
 Y_train_label{val}=Y_test_vali{val};
 
%% 训练模型(监督学习)
iters=maxIter;
epsilon =0.0001;

%%第一种情况
for v=1:num_view
X_train_vali_v11{val}{v}=zeros(size(X_train_vali{val}{v},1),num_class);
for i=1:length(Y_train_vali{val})
    X_train_vali_v11{val}{v}(i,Y_train_vali{val}(i))=1;
end
XX_train_vali{val}{v}=[X_train_vali_v11{val}{v};X_train_feature{val}{v}];
YY=[Y_train_vali{val};Y_train_label{val}]; %%第一个视角表示第一次5折的测试标签，第二个视角表示第二次5折测试标签。
end

  
 [W{val},M{val},B{val},Objective{val}]=trainlsr1(XX_train_vali{val},YY,iters, epsilon,maxIter);%% 用所有的训练集没有部分训练集画的曲线好

%% 测试模型
test_idex1= list_test;
%*************矩阵转换***********************
for v=1:num_view
X_train_vali_v1{v,1}=X_train{v};
X_test_vali_v1{v,1}=X_test{v};
end

%**************元分类器测试数据***********(SVM学习的模型)(方法一)
 for v=1:num_view
  [Test_results_M{1,val}{v},accuracy, dec_values] = libsvmpredict( Y_test,X_test_vali_v1{v},model{val}{v});
 end
 
%*************将KNN用于测试数据***********************（方法二）
%  for v=1:num_view
%       X_train_vali_v2{1,val}(v)=X_train_vali_v1(v);
%       X_test_vali_v2{1,val}(v)=X_test_vali_v1(v);
%  end   
%  Test_results_M{1,val}=knn_classify(X_train_vali_v2{val},X_test_vali_v2{val},kn,Y_train,num_view);

%*************决策树用于测试数据********************** (方法三）
%    for v=1:num_view
%    Test_results_M{1,val}{v} = predict(tree{val}{v},full(X_test_vali_v1{v}));
%    end


% *********************01特征形式**************************
test_num_sum=test_idex1;
 for v=1:num_view
        X_TEST_FEATURE{val}{v}=zeros(test_num_sum,num_class);
    for i=1:test_num_sum
        X_TEST_FEATURE{val}{v}(i,Test_results_M{val}{v}(i))=1;
    end
 end
 X_test_vali{val}=X_TEST_FEATURE{val};
 
% ***********每个视角的分类结果给一个权重*****************
for iter=1:maxIter
sumvote{val}{iter}=0;
for v=1:num_view
      test_data{v}=X_test_vali{val}{v}*(W{val}{iter}{v})';
      sumvote{val}{iter}=sumvote{val}{iter}+test_data{v};
end
[maxvalue{val},idx{val}{iter}]=max(sumvote{val}{iter}');
% %% 测试集上误差计算
Y_test1=zeros(length(Y_test),num_class);
for i=1:length(Y_test)
    Y_test1(i,Y_test(i))=1;
end
Testerror=0;
for v=1:num_view
    P=X_test_vali{val}{v}*W{val}{iter}{v}-Y_test1;
    Testerror=Testerror+sqrt(trace(P'*P));
end
TestError{val}(iter)=Testerror;
%% 分类精度分析
result{val}{iter}=Classifymeasure(idx{val}{iter},Y_test);
end
   end 
  for iter1=1:maxIter
     acc=[];
     for val=1:1
     acc=[result{val}{iter1}(1);acc];
     end
    [value,order]=max(acc);
     Test_Result_error(iter1)=TestError{order}(iter1);
     Train_Result_error(iter1)=Objective{order}(iter1);
     Result(iter1)=result{order}(iter1);
  end




            
            
