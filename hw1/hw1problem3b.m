syms b
load HW1_3.mat;

w = inv(X'*X)*X'*Y;

for k=1:100
    line([k k],[0 w(k)])
end
squrederror = 0;
for i=1:n
    squrederror = squrederror + (Y(i)-X(i,:)*w)^2;
end
mse = squrederror/n;
hold on
Yhat = X*w;
bias = abs(sum(Y)/n-sum(Yhat)/n);
squarevariance = 0;
yhatbar = sum(Yhat)/n;
for i=1:n
    squarevariance  = squarevariance + Yhat(i)^2-yhatbar^2;
end
variance = squarevariance/(n-1);
fprintf("mean squre error:%x\n",mse)
fprintf("|bias|:%x\n",bias)
fprintf("variance:%x\n",variance)

for k=1:100
    plot(k,w(k),"o")
end

xlabel('model weight index','FontSize',12);
ylabel('model weight','FontSize',12);
hold off
