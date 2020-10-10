function [W1,M,B,Objective]=trainlsr1(Xl,Yl, iters, epsilon,maxIter);

[N, dim] = size(Xl{1});
num_class = max(Yl);
view=length(Xl);
class_id=Yl;
train_length=length(Yl);
% class_d=-1*ones(train_length,num_class);
class_d=zeros(train_length,num_class);
for i=1:train_length
    class_d(i,Yl(i))=1;
end
Y=class_d;
M = zeros(num_class,N);
B = -1 * ones(N, num_class);

for i = 1 : N
   M( class_id(i),i ) = 1.0;  
   B(i, class_id(i) )  = 1.0;
end
%% initialization

XX_train = []; XX_test = [];
dv = zeros(view, 1);
dim = zeros(view, 1);

flag = 0;
for i = 1 : view
    [dim(i), N] = size(Xl{i}'); % N is the number of samples in the training set
    dv(i) = 1 / view;
    flag = flag + dim(i);
end

dv_dim = []; flag=0;

for v = 1:view
    dv_dim((flag+1):(flag + dim(v)),:) = repmat(dv(v),dim(v),1);
    viewDim(v) = size(Xl{v},2);
    [W{v}, b{v}] = new_least_squares_regression(Xl{v}', Y');%%参数调节
    alpha{v}=1/view;
end


%% updating
for iter = 1: iters
    Q=zeros(size(Y'));
    a=0;
    %first, optimize matrix M.
    for v=1:view
        P{v}=W{v}'*Xl{v}'-Y';
        Q=Q+alpha{v}*P{v};
        a=a+alpha{v};
    end
    M = optimize_m_matrix((Q/a)', B);
    % second, optimize matrix W.

    for v=1:view
        R  = Y + (B .* M);
        W{v}=pinv(Xl{v}'*Xl{v})*Xl{v}'*R; %% 直接求逆，效果差，伪逆效果较好
    end
%     
%     % third, optimize matrix \alpha.
    for v=1:view
    T{v}=(norm(W{v}'*Xl{v}'-R', 'fro'));
    alpha{v}=1/(2*T{v});
    end
    % fourth, computing objective
    obj=0;
    for v=1:view
       obj=obj+T{v};
    end
    Objective(iter)=obj;

  %% 训练误差(新增加)
W1{iter}=W;
end

