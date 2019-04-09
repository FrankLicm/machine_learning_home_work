clc
N = 10;
K = 2;
Sigma1 = [2 -1;-1 1];
mu1 = [2 2];
remu1=repmat(mu1,N,1);
[n1,d1]=size(remu1);
[E1,Lambda1]=eig(Sigma1);
U1 = sqrt(Lambda1)*E1';
X1 = randn(n1,d1)*U1 + remu1;
scatter(X1(:,1),X1(:,2),50,"bo");
hold on
Sigma2 = [1,0.5;0.5 1];
mu2 = [0 0];
remu2=repmat(mu2,N,1);
[n2,d2]=size(remu2);
[E2,Lambda2]=eig(Sigma2);
U2 = sqrt(Lambda2)*E2';
X2 = randn(n2,d2)*U2 + remu2;
scatter(X2(:,1),X2(:,2),50,"ro");

%QDA decision boundary
emu1 = sum(X1)/ N;
emu2 = sum(X2)/ N;
esum1 = [0 0;0 0]; 
for k=1:length(X1)
   esum1 = esum1+(X1(k,:)-emu1)'*(X1(k,:)-emu1);
end
esum2 = [0 0;0 0];
for k=1:length(X2)
   esum2 = esum2+(X2(k,:)-emu2)'*(X2(k,:)-emu2);
end
esigma1 = esum1/(N-1);
esigma2 = esum2/(N-1);
delta = @(x1,x2) -1/2*log(det(esigma1))-1/2*([x1 x2] - emu1)*inv(esigma1)*([x1 x2]-emu1)'+1/2*log(det(esigma2))+1/2*([x1 x2] - emu2)*inv(esigma2)*([x1 x2]-emu2)';
fimplicit(delta,"g");


% Theoretical Bayes decision boundary
G = @(a,b)(2*pi)^(-1)*abs(det(Sigma1))^(-1/2)*exp((-1/2)*([a b]-mu1)*inv(Sigma1)*([a b]-mu1)') - (2*pi)^(-1)*abs(det(Sigma2))^(-1/2)*exp((-1/2)*([a b]-mu2)*inv(Sigma2)*([a b]-mu2)');
fimplicit(G,"k-");

% Empirical Bayes decision boundary
emu1 = sum(X1)/ N;
emu2 = sum(X2)/ N;
esum1 = [0 0;0 0]; 
for k=1:length(X1)
   esum1 = esum1+(X1(k,:)-emu1)'*(X1(k,:)-emu1);
end
esum2 = [0 0;0 0];
for k=1:length(X2)
   esum2 = esum2+(X2(k,:)-emu2)'*(X2(k,:)-emu2);
end
esigma1 = esum1/(N-1);
esigma2 = esum2/(N-1);
G = @(a,b)(2*pi)^(-1)*abs(det(esigma1))^(-1/2)*exp((-1/2)*([a b]-emu1)*inv(esigma1)*([a b]-emu1)') - (2*pi)^(-1)*abs(det(esigma2))^(-1/2)*exp((-1/2)*([a b]-emu2)*inv(esigma2)*([a b]-emu2)');
fimplicit(G,"k--");

legend("sample1","sample2","QDA","Bayes(theoretical)","Bayes(estimated)",'Location','northwest')
xlabel('x1','FontSize',12);
ylabel('x2','FontSize',12);
hold off