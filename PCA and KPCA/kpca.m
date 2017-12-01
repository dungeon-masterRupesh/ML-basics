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
%% above part of reading from data is written by a friend Sushant Tarun
%% lets start with a kernel maybe polynomial for step 1
%% k(x,y) = (x.y)^2

%% doing it for 1 letter let for numeral 1 i.e. A1 is matrix

%% Here we have c1 samples. Our resulting will be a linear combination of phi(xi)
%% all the c1 samples

%% i.e. let V be the eigen vector. So V = ai * phi(x1)

%% a1 = zeros(c1,1);

%% so we have to minimize V*x conditioned ||V|| = 1

%% centralise the data to it's mean
A1 = A1 - repmat(mean(1,:)',1,116)';
%% Using gram matrix
K = zeros(c1,c1);
for i = 1:c1
    for j = i:c1
        %% calculating k(x(i),x(j))
        ans = 0;
        for r = 1:784
            ans = ans + A1(i,r)*A1(j,r);
        end
        ans  = ans * ans;
        K(i,j) = ans;
        K(j,i) = ans;
    end
end
%% centralising K
K = K - (zeros(116)+(1/116)) * K - K * (zeros(116)+(1/116)) + (zeros(116)+(1/116)) * K * (zeros(116)+(1/116));
%% eigen decomposition
[a1 , e1 ] = eig(K);
a1max = a1(:,116);
%% value of ai for xi
%% v = a1max . phi(x)
%% normalising V so a1max.a1max = 1/eigenvalue
a1max = a1max / sqrt(e1(116,116));

%% this was done using eigen value of matrix
%% other way is to find a that maximizes variance of data along V. 



