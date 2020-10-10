function [W,M,Test_Result_error,Result,Train_Result_error]=EWMC_classification(X_train,Y_train,X_test,Y_test,each_class_num,list_train,list_test,num_class,num_view,maxIter)
%Title:When Multi-view Classification Meets Ensemble Learning

%% 5�۽�����֤��ѡ��ѵ�����Ͳ��Լ�  
    subtrain_num=list_train;
    sub_each_class_num=floor(subtrain_num/num_class);
    ix=sub_each_class_num;
    % ��ֹ����ĳһ��Ϊ��
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

%% ����ת��
   if length(X_train_vali)>1
      for v=1:num_view
         X_train_vali_v{val}{v,1}=X_train_vali{val}{1,v};
         X_test_vali_v{val}{v,1}=X_test_vali{val}{1,v};
      end
  else     
     X_train_vali_v=X_train_vali;
     X_test_vali_v=X_test_vali;
   end
  
%% (Multi-class SVM��ѧϰ����������������һ��
  for v=1:num_view
  model{val}{v}= libsvmtrain(Y_train_vali{val}, X_train_vali_v{val}{v}, '-s 0 -t 0 -c 1');
  [results_M{1,val}{v}, accuracy, dec_values] = libsvmpredict( Y_test_vali{val}, X_test_vali_v{val}{v}, model{val}{v});
  end
  
%% (KNN)ѧϰ���������� ����������
% kn=5;
% results_M{1,val} = knn_classify(X_train_vali_v{val},X_test_vali_v{val},kn,Y_train_vali{val}',num_view);

%% (������)ѧϰ����������(������)
%      for v=1:num_view
%      tree{val}{v} = fitctree(full(X_train_vali_v{val}{v}),Y_train_vali{val},'Prune','off');
%      results_M{1,val}{v} = predict(tree{val}{v},full(X_test_vali_v{val}{v}));
%      end
  

%% ����ת��
 for v=1:num_view
     strain_number=length(results_M{val}{v});
      X_train_feature{val}{v}=zeros(strain_number,num_class);
      for i=1:strain_number   
      X_train_feature{val}{v}(i,results_M{val}{v}(i))=1;
      end  
 end
 Y_train_label{val}=Y_test_vali{val};
 
%% ѵ��ģ��(�ලѧϰ)
iters=maxIter;
epsilon =0.0001;

%%��һ�����
for v=1:num_view
X_train_vali_v11{val}{v}=zeros(size(X_train_vali{val}{v},1),num_class);
for i=1:length(Y_train_vali{val})
    X_train_vali_v11{val}{v}(i,Y_train_vali{val}(i))=1;
end
XX_train_vali{val}{v}=[X_train_vali_v11{val}{v};X_train_feature{val}{v}];
YY=[Y_train_vali{val};Y_train_label{val}]; %%��һ���ӽǱ�ʾ��һ��5�۵Ĳ��Ա�ǩ���ڶ����ӽǱ�ʾ�ڶ���5�۲��Ա�ǩ��
end

  
 [W{val},M{val},B{val},Objective{val}]=trainlsr1(XX_train_vali{val},YY,iters, epsilon,maxIter);%% �����е�ѵ����û�в���ѵ�����������ߺ�

%% ����ģ��
test_idex1= list_test;
%*************����ת��***********************
for v=1:num_view
X_train_vali_v1{v,1}=X_train{v};
X_test_vali_v1{v,1}=X_test{v};
end

%**************Ԫ��������������***********(SVMѧϰ��ģ��)(����һ)
 for v=1:num_view
  [Test_results_M{1,val}{v},accuracy, dec_values] = libsvmpredict( Y_test,X_test_vali_v1{v},model{val}{v});
 end
 
%*************��KNN���ڲ�������***********************����������
%  for v=1:num_view
%       X_train_vali_v2{1,val}(v)=X_train_vali_v1(v);
%       X_test_vali_v2{1,val}(v)=X_test_vali_v1(v);
%  end   
%  Test_results_M{1,val}=knn_classify(X_train_vali_v2{val},X_test_vali_v2{val},kn,Y_train,num_view);

%*************���������ڲ�������********************** (��������
%    for v=1:num_view
%    Test_results_M{1,val}{v} = predict(tree{val}{v},full(X_test_vali_v1{v}));
%    end


% *********************01������ʽ**************************
test_num_sum=test_idex1;
 for v=1:num_view
        X_TEST_FEATURE{val}{v}=zeros(test_num_sum,num_class);
    for i=1:test_num_sum
        X_TEST_FEATURE{val}{v}(i,Test_results_M{val}{v}(i))=1;
    end
 end
 X_test_vali{val}=X_TEST_FEATURE{val};
 
% ***********ÿ���ӽǵķ�������һ��Ȩ��*****************
for iter=1:maxIter
sumvote{val}{iter}=0;
for v=1:num_view
      test_data{v}=X_test_vali{val}{v}*(W{val}{iter}{v})';
      sumvote{val}{iter}=sumvote{val}{iter}+test_data{v};
end
[maxvalue{val},idx{val}{iter}]=max(sumvote{val}{iter}');
% %% ���Լ���������
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
%% ���ྫ�ȷ���
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




            
            
