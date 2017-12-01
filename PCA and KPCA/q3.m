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

cov0=cov(A0);
cov1=cov(A1);
cov2=cov(A2);
cov3=cov(A3);
cov4=cov(A4);
cov5=cov(A5);
cov6=cov(A6);
cov7=cov(A7);
cov8=cov(A8);
cov9=cov(A9);

[V0, E0]=eig(cov0);
[V1, E1]=eig(cov1);
[V2, E2]=eig(cov2);
[V3, E3]=eig(cov3);
[V4, E4]=eig(cov4);
[V5, E5]=eig(cov5);
[V6, E6]=eig(cov6);
[V7, E7]=eig(cov7);
[V8, E8]=eig(cov8);
[V9, E9]=eig(cov8);

e0=eig(cov0);
figure;
plot(e0);
saveas(gcf, 'eigen0.png');

e1=eig(cov1);
figure;
plot(e1);
saveas(gcf, 'eigen1.png');

e2=eig(cov2);
figure;
plot(e2);
saveas(gcf, 'eigen2.png');

e3=eig(cov3);
figure;
plot(e3);
saveas(gcf, 'eigen3.png');

e4=eig(cov4);
figure;
plot(e4);
saveas(gcf, 'eigen4.png');

e5=eig(cov5);
figure;
plot(e5);
saveas(gcf, 'eigen5.png');

e6=eig(cov6);
figure;
plot(e6);
saveas(gcf, 'eigen6.png');

e7=eig(cov7);
figure;
plot(e7);
saveas(gcf, 'eigen7.png');

e8=eig(cov8);
figure;
plot(e8);
saveas(gcf, 'eigen8.png');

e9=eig(cov9);
figure;
plot(e9);
saveas(gcf, 'eigen9.png');

mu_minus_0=reshape(mean(1,:),784,1)-sqrt(e0(784))*reshape(V0(:,784),784,1);
mu_plus_0=reshape(mean(1,:),784,1)+sqrt(e0(784))*reshape(V0(:,784),784,1);
figure;
subplot(1,3,1);
imagesc(reshape(mu_minus_0, 28, 28));
subplot(1,3,2);
imagesc(reshape(mean(1,:),28,28));
subplot(1,3,3);
imagesc(reshape(mu_plus_0, 28, 28));
saveas(gcf, 'image0.png');

mu_minus_1=reshape(mean(2,:),784,1)-sqrt(e1(784))*reshape(V1(:,784),784,1);
mu_plus_1=reshape(mean(2,:),784,1)+sqrt(e1(784))*reshape(V1(:,784),784,1);
figure;
subplot(1,3,1);
imagesc(reshape(mu_minus_1, 28, 28));
subplot(1,3,2);
imagesc(reshape(mean(2,:),28,28));
subplot(1,3,3);
imagesc(reshape(mu_plus_1, 28, 28));
saveas(gcf, 'image1.png');

mu_minus_2=reshape(mean(3,:),784,1)-sqrt(e2(784))*reshape(V2(:,784),784,1);
mu_plus_2=reshape(mean(3,:),784,1)+sqrt(e2(784))*reshape(V2(:,784),784,1);
figure;
subplot(1,3,1);
imagesc(reshape(mu_minus_2, 28, 28));
subplot(1,3,2);
imagesc(reshape(mean(3,:),28,28));
subplot(1,3,3);
imagesc(reshape(mu_plus_2, 28, 28));
saveas(gcf, 'image2.png');

mu_minus_3=reshape(mean(4,:),784,1)-sqrt(e3(784))*reshape(V3(:,784),784,1);
mu_plus_3=reshape(mean(4,:),784,1)+sqrt(e3(784))*reshape(V3(:,784),784,1);
figure;
subplot(1,3,1);
imagesc(reshape(mu_minus_3, 28, 28));
subplot(1,3,2);
imagesc(reshape(mean(4,:),28,28));
subplot(1,3,3);
imagesc(reshape(mu_plus_3, 28, 28));
saveas(gcf, 'image3.png');

mu_minus_4=reshape(mean(5,:),784,1)-sqrt(e4(784))*reshape(V4(:,784),784,1);
mu_plus_4=reshape(mean(5,:),784,1)+sqrt(e4(784))*reshape(V4(:,784),784,1);
figure;
subplot(1,3,1);
imagesc(reshape(mu_minus_4, 28, 28));
subplot(1,3,2);
imagesc(reshape(mean(5,:),28,28));
subplot(1,3,3);
imagesc(reshape(mu_plus_4, 28, 28));
saveas(gcf, 'image4.png');

mu_minus_5=reshape(mean(6,:),784,1)-sqrt(e5(784))*reshape(V5(:,784),784,1);
mu_plus_5=reshape(mean(6,:),784,1)+sqrt(e5(784))*reshape(V5(:,784),784,1);
figure;
subplot(1,3,1);
imagesc(reshape(mu_minus_5, 28, 28));
subplot(1,3,2);
imagesc(reshape(mean(6,:),28,28));
subplot(1,3,3);
imagesc(reshape(mu_plus_5, 28, 28));
saveas(gcf, 'image5.png');

mu_minus_6=reshape(mean(7,:),784,1)-sqrt(e6(784))*reshape(V6(:,784),784,1);
mu_plus_6=reshape(mean(7,:),784,1)+sqrt(e6(784))*reshape(V6(:,784),784,1);
figure;
subplot(1,3,1);
imagesc(reshape(mu_minus_6, 28, 28));
subplot(1,3,2);
imagesc(reshape(mean(7,:),28,28));
subplot(1,3,3);
imagesc(reshape(mu_plus_6, 28, 28));
saveas(gcf, 'image6.png');

mu_minus_7=reshape(mean(8,:),784,1)-sqrt(e7(784))*reshape(V7(:,784),784,1);
mu_plus_7=reshape(mean(8,:),784,1)+sqrt(e7(784))*reshape(V7(:,784),784,1);
figure;
subplot(1,3,1);
imagesc(reshape(mu_minus_7, 28, 28));
subplot(1,3,2);
imagesc(reshape(mean(8,:),28,28));
subplot(1,3,3);
imagesc(reshape(mu_plus_7, 28, 28));
saveas(gcf, 'image7.png');

mu_minus_8=reshape(mean(9,:),784,1)-sqrt(e8(784))*reshape(V8(:,784),784,1);
mu_plus_8=reshape(mean(9,:),784,1)+sqrt(e8(784))*reshape(V8(:,784),784,1);
figure;
subplot(1,3,1);
imagesc(reshape(mu_minus_8, 28, 28));
subplot(1,3,2);
imagesc(reshape(mean(9,:),28,28));
subplot(1,3,3);
imagesc(reshape(mu_plus_8, 28, 28));
saveas(gcf, 'image8.png');

mu_minus_9=reshape(mean(10,:),784,1)-sqrt(e9(784))*reshape(V9(:,784),784,1);
mu_plus_9=reshape(mean(10,:),784,1)+sqrt(e9(784))*reshape(V9(:,784),784,1);
figure;
subplot(1,3,1);
imagesc(reshape(mu_minus_9, 28, 28));
subplot(1,3,2);
imagesc(reshape(mean(10,:),28,28));
subplot(1,3,3);
imagesc(reshape(mu_plus_9, 28, 28));
saveas(gcf, 'image9.png');