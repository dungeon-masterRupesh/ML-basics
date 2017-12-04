%   y: dimensionanlity-reduced data
%	eigVector: eigen-vector obtained in kPCA
%   X: data matrix
%   para: parameter of Gaussian kernel
%	z: pre-image of y

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its 
%   Applications in Face Recognition and Active Shape Models. 
%   arXiv:1207.3538 [cs.CV], 2012. 

function Z=kPCA_PreImage(y,eigVector,X,para)

iter=1000;
N=size(X,1);
d=size(y,2);

% gamma=zeros(1,N);
% for i=1:N
%     gamma(i)=eigVector(i,1:d)*y(i,1:d)';
% end
gamma_all = y * eigVector(:,1:d)';
Z = zeros(N,size(X,2));
for i = 1:N
gamma = gamma_all(i,:);
z=mean(X)'; % initialization
for count=1:iter
    pre_z=z;
    xx=bsxfun(@minus,X',z);
    xx=xx.^2;
    xx=-sum(xx)/(2*para.^2);
    xx=exp(xx);
    xx=xx.*gamma;
    
    z=xx*X/sum(xx);
    z=z';
    if norm(pre_z-z)/norm(z)<0.00001
        break;
    end
end
Z(i,:)=z';
end
end
