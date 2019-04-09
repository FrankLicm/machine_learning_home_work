syms x
load HW1_4.mat;
p = polynomial(X,Y,3);
f1 = @(x) alpha(1)*x^3 + alpha(2)*x^2 + alpha(3)*x^1+alpha(4);
f2 = @(x) p(1)*x^3 + p(2)*x^2 + p(3)*x^1+ p(4);

scatter(X,Y,50,"k.");
hold on
fplot(f1,[-1,1],"b")
fplot(f2,[-1,1],"r")
xlabel('x','FontSize',12);
hold off
legend("samples"," true model","estimated model");

function p = polynomial(x, y, n)
x = x(:);
V = ones(length(x), n + 1);
for j = n:-1:1
   V(:, j) = V(:, j + 1) .* x;
end
[Q, R] = qr(V, 0);
p      = (R \ (Q' * y(:)))';
end