close all; 
clear;
d = 2;
k = 3;
n = 10000;
[X,label] = toyexample(d,k,n);
[label2,mu] = kmeans(X,k);
figure;
plotc(X,label2, mu');

function [label, mu] = kmeans(X, m)
[~,n] = size(X);
mu = X(:,randperm(n,m));
[~,label] = min(dot(mu,mu,1)'/2-mu'*X,[],1); 
n = numel(label);
idx = 1:n;
last = zeros(1,n);
while any(label ~= last)
    [~,~,last(:)] = unique(label);                 
    mu = X*normalize(sparse(idx,last,1),1);         
    [~,label] = min(dot(mu,mu,1)'/2-mu'*X,[],1);  
end
end

function Y = normalize(X, dim)
if nargin == 1
    dim = find(size(X)~=1,1);
    if isempty(dim), dim = 1; end
end
Y = X./sum(X,dim);
end

function plotc(X, label, center)
[~,n] = size(X);
if nargin == 1
    label = ones(n,1);
end
assert(n == length(label));
color = 'brgmcyk';
m = length(color);
c = max(label);
figure(gcf);
clf;
hold on;
view(2);
for i = 1:c
   idc = label==i;
   scatter(X(1,idc),X(2,idc),36,color(mod(i-1,m)+1),".");
   plot(center(i,1),center(i,2),"k-s")
end
xlabel('X_1')
ylabel('X_2')
axis equal
grid on
hold off
end

function [X, z, mu] = toyexample(d, k, n)
alpha = 1;
beta = nthroot(k,d); 
X = randn(d,n);
w = dirichletRnd(alpha,ones(1,k)/k);
z = discreteRnd(w,n);
E = full(sparse(z,1:n,1,k,n,n));
mu = randn(d,k)*beta;
X = X+mu*E;
end

function x = dirichletRnd(a, m)
if nargin == 2
    a = a*m;
end
x = gamrnd(a,1);
x = x/sum(x);
end

function x = discreteRnd(p, n)
if nargin == 1
    n = 1;
end
r = rand(1,n);
p = cumsum(p(:));
[~,x] = histc(r,[0;p/p(end)]);
end









