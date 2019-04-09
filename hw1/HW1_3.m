close all; clear all; clc;

% %generate data
N = 90;
p = 500;
nonzero = 5;
power = 1e-3;
X = rand(N, p);
[~, random] = sort(rand(p,1));
select = random(1:nonzero);
beta = zeros(p,1);
beta(select) = 1;
Y = X * beta + power * randn(N, 1);
X_test = rand(N, p);
Y_test = X_test * beta + power * randn(N, 1);

%load data
%load HW1_3.mat;

%optimization parameters
lambda = 1e-1
gamma = 5e-3;
iterations = 10000;

%get data dimensions
N = size(X,1);
p = size(X,2);

%initalize array for solution trajectory
beta_hat = zeros(p, iterations);
beta_hat(:,1) = rand(p, 1);

soft_thresh = @(beta_hat, tau) beta_hat .* (...
    (((beta_hat - tau) ./ beta_hat).*(beta_hat >= tau)) + ...
     ((beta_hat + tau) ./ beta_hat).*(beta_hat <= -tau))
 
for i = 2:iterations

    %calculate portion of gradient due to model error
    error_grad = -2/N * ((Y - X * beta_hat(:, i-1)).' * X).';
    
    %update model
    beta_hat(:,i) = soft_thresh(beta_hat(:,i-1) - gamma * error_grad,...
        gamma * lambda);

end

%least squares - inverse
beta_hat_ls = inv(X.'*X)*X.'*Y;

%least squares - pseudo-inverse
beta_hat_lspi = pinv(X.'*X)*X.'*Y;

%calculate least squares error
LASSO_error = sum((Y_test-X_test*beta_hat(:,end)).^2) / length(Y_test)
LS_error = sum((Y_test-X_test*beta_hat_ls).^2) / length(Y_test)
LSPI_error = sum((Y_test-X_test*beta_hat_lspi).^2) / length(Y_test)


%display LASSO solution
figure;
subplot(3,1,1);
stem(beta_hat(:,end), 'b');
xlabel('model weight index');
ylabel('model weight')
title(['LASSO - MSE = ' num2str(LASSO_error)]);
ylim(1.1*[min(beta_hat(:,end)) max(beta_hat(:,end))]);

%display least squares inv solution
subplot(3,1,2);
stem(beta_hat_ls, 'b');
xlabel('model weight index');
ylabel('model weight');
title(['Least squares - inverse - MSE = ' num2str(LS_error)]);
ylim(1.1*[min(beta_hat_ls) max(beta_hat_ls)]);

%display least squares pseudo-inv solution
subplot(3,1,3);
stem(beta_hat_lspi, 'b');
xlabel('model weight index');
ylabel('model weight');
title(['Least squares - pseud-inverse - MSE = ' num2str(LSPI_error)]);
ylim(1.1*[min(beta_hat_lspi) max(beta_hat_lspi)]);


