Sigma = [1 -0.5;-0.5 0.5];
mu = [1 1];
mu2=repmat(mu,1000,1);
[n,d]=size(mu2);
[E,Lambda]=eig(Sigma);
U = sqrt(Lambda)*E';
X = randn(n,d)*U + mu2;
scatter(X(:,1),X(:,2),50,"k.");
hold on
xy1 = E(:,1);
xy2 = E(:,2);
quiver(1,1,xy1(1),xy1(2),"-r",'LineWidth',2.5)
quiver(1,1,xy2(1),xy2(2),"-r",'LineWidth',2.5)

x = -3:.1:5; 
y = -3:.1:5; 
[X,Y] = meshgrid(x,y); 
G = @(a,b)(2*pi)^(-1)*abs(det(Sigma))^(-1/2)*exp((-1/2)*([a b]-mu)*inv(Sigma)*([a b]-mu)');
fcontour(G,[-3 5 -3 5],'LineWidth',1.5)
legend("samples","eigenvector 1","eigenvector 2","pdf");
xlabel('x1','FontSize',12);
ylabel('x2','FontSize',12);
hold off