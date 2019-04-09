syms x y
load HW1_4.mat;
scatter(X,Y,50,"k.");
hold on
sampleSize = 5;
maxDistance = 0.3; 
points = [X Y];
iterations = 10000;
n_inliners = 0.4 * 1000;
[p, inlierPtrs] = polynomialransac(points,sampleSize,maxDistance,iterations,n_inliners);

f1 = @(x) alpha(1)*x^3 + alpha(2)*x^2 + alpha(3)*x^1+alpha(4);
f2 = @(x) p(1)*x^3 + p(2)*x^2 + p(3)*x^1+ p(4);
scatter(inlierPts(:,1),inlierPts(:,2),50,"r.");
fplot(f1,[-1,1],"b")
fplot(f2,[-1,1],"g")
xlabel('x','FontSize',12);
hold off
legend("samples"," consensus set"," true model","estimated model");

function p = polynomial(x, y, n)
x = x(:);
V = ones(length(x), n + 1);
for j = n:-1:1
   V(:, j) = V(:, j + 1) .* x;
end
[Q, R] = qr(V, 0);
p      = (R \ (Q' * y(:)))';
end


function [p,Inliers] = polynomialransac(points,sampleSize,maxDistance,iterations,n_inliners)
p = [];
Inliers = [];
[N,~] = size(points);
for i=1:iterations
    inliers = [];
    n_i = 0;    
    idx=randperm(N);
    idx=idx(1:sampleSize);
    samples=points(idx,:);
    w = polynomial(samples(:,1), samples(:,2), 3);
    for k=1:N
        distance = abs(points(k, 2) - polyval(w, points(k,1)));
        
        if( distance < maxDistance )
            inliers(n_i+1,:) = points(k,:);
            n_i = n_i+1;
        end
    end    
    if ( n_inliners < n_i )
        p = w;
        Inliers = inliers;
        break
    end   
end
end
