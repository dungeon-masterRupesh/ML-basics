load('ex6data2.mat');
plotData(X,y)
for c=[1 2 3 5 7 10 12 15 20]
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    [Y, eigVector, eigValue]=kPCA(X,5,'gaussian',c);
    figure;
    z=kPCA_PreImage(Y,eigVector,X,c);
    plotData(z,y)
    xlabel(c);
end