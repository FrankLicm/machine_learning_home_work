load HW1_3.mat;
w = lasso(X,Y,1);
for k=1:100
    line([k k],[0 w(k)])
end
hold on

for k=1:100
    if w(k)>0.8
        text(k-1,w(k)+0.03,""+k)
    end
    plot(k,w(k),"o")
end
% output mean squre error
[n,~] = size(Y);
squrederror = 0;
for i=1:n
    squrederror = squrederror + (Y(i)-X(i,:)*w)^2;
end
Yhat = X*w;
mse = squrederror/n;
bias = sum(Y)/n-sum(Yhat)/n;
squarevariance = 0;
yhatbar = sum(Yhat)/n;

for i=1:n
    squarevariance  = squarevariance + Yhat(i)^2-yhatbar^2;
end
variance = squarevariance/(n-1);
fprintf("mean squre error:%x\n",mse)
fprintf("|bias|:%x\n",bias)
fprintf("variance:%x\n",variance)
xlabel('model weight index','FontSize',12);
ylabel('model weight','FontSize',12);
hold off
function x=lasso(X,y,lambda)
[~,n2] = size(X);
x = zeros(n2,1);
k=1;
t = 0.0005;
while k<10000
    s = x;
    for i=1:n2
        if s(i)>lambda
            s(i) = 1;
        elseif s(i)< -lambda
            s(i) = -1;
        else
            s(i) = 0;
        end
    end
    g = X'*(X*x-y)+lambda*s;
    x = x - t*g;
    k = k+1;
end
end

