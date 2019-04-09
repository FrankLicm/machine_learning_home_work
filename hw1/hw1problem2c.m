Sigma = [1 -0.5;-0.5 0.5];
mu = [1 1];
mu2=repmat(mu,1000,1);
[n,d]=size(mu2);
[E,Lambda]=eig(Sigma);
U = sqrt(Lambda)*E';
X = randn(n,d)*U + mu2;
scatter(X(:,1),X(:,2),50,"k.");
hold on


a = -3:.1:5; 
b = -3:.1:5; 
syms a b
[X,Y] = meshgrid(x,y); 
D = @(a,b)norm([a b]-mu);
fcontour(D,[-3 5 -3 5],'LineWidth',1.5)
legend("samples","Euclidean distance");
xlabel('x1','FontSize',12);
ylabel('x2','FontSize',12);
hold off