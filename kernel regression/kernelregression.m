clc; clear;
images=loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
N=1000;
mean=zeros(10, 784);
sum=zeros(10, 784);
occur=zeros(10);

for i=1:N
    sum(labels(i)+1,:) = sum(labels(i)+1,:)+ images(:,i).';
    occur(labels(i)+1)=occur(labels(i)+1)+1;
end
A0=zeros(occur(1),784);
A1=zeros(occur(2),784);
A2=zeros(occur(3),784);
A3=zeros(occur(4),784);
A4=zeros(occur(5),784);
A5=zeros(occur(6),784);
A6=zeros(occur(7),784);
A7=zeros(occur(8),784);
A8=zeros(occur(9),784);
A9=zeros(occur(10),784);
for i=1:10
    mean(i, :)=sum(i, :)/occur(i);
end

c0=0;
c1=0;
c2=0;
c3=0;
c4=0;
c5=0;
c6=0;
c7=0;
c8=0;
c9=0;

for i=1:N
    if(labels(i)==0)
        c0=c0+1;
        A0(c0,:)=images(:,i);
    end
    if(labels(i)==1)
        c1=c1+1;
        A1(c1,:)=images(:,i);
    end
    if(labels(i)==2)
        c2=c2+1;
        A2(c2,:)=images(:,i);
    end
    if(labels(i)==3)
        c3=c3+1;
        A3(c3,:)=images(:,i);
    end
    if(labels(i)==4)
        c4=c4+1;
        A4(c4,:)=images(:,i);
    end
    if(labels(i)==5)
        c5=c5+1;
        A5(c5,:)=images(:,i);
    end
    if(labels(i)==6)
        c6=c6+1;
        A6(c6,:)=images(:,i);
    end
    if(labels(i)==7)
        c7=c7+1;
        A7(c7,:)=images(:,i);
    end
    if(labels(i)==8)
        c8=c8+1;
        A8(c8,:)=images(:,i);
    end
    if(labels(i)==9)
        c9=c9+1;
        A9(c9,:)=images(:,i);
    end
end
%% center both A1 and A2 in original space Comparing 5 and 2
A1 = A5 - repmat(mean(5,:)',1,c5)';
A2 = A2 - repmat(mean(1,:)',1,c2)';

%% lets do an experiment by taking 50 samples from 1 and 50 samples from 1 and 2.
Aex = [A1(1:50,:);A2(1:50,:)];
sigma = 0.005;
siz = 100;
%% Using gram matrix
K = zeros(siz,siz);
for i = 1:siz
    for j = i:siz
        %% calculating k(x(i),x(j))
        val = norm(Aex(i,:)-Aex(j,:));
        val  = val * val;
        val = exp(-1*val*sigma);
        K(i,j) = val;
        K(j,i) = val;
    end
end
%% centralising K
K = K - (zeros(siz)+(1/siz)) * K - K * (zeros(siz)+(1/siz)) + (zeros(siz)+(1/siz)) * K * (zeros(siz)+(1/siz));
%% make the result y
yex  = [ones(50,1);ones(50,1).*(-1)];

%% assume a value of gamma for regression
gam = 1;
%% solve (siz * gam * Isiz + K) * c = y;
lhs = siz * gam * eye(siz) + K;
c = pinv(lhs)* yex; 
Atest = [A1(51:60,:);A2(51:60,:)];
Ktt= zeros(siz,20);
for i = 1:20
    for j = 1:siz
        %% calculating k(x(i),x(j))
        val = norm(Atest(i,:)-Aex(j,:));
        val  = val * val;
        val = exp(-1*val*sigma);
        Ktt(j,i) = val;
    end
end
Ktt = Ktt';
resul = sign(Ktt * c);