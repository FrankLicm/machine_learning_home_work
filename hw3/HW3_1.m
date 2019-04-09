close all; clear all; clc;

% %generate data
K = 5;
alpha = 1;
lambdas = logspace(-5, 2, 500);

%load data and preprocess
load ionosphere.mat;
y = double(strcmp(Y, 'g'));
N = size(X,1);
p = size(X,2);

%generate outer fold indices
outer = mod(1:N, K)+1;

%define blocking function to balance classes
block = @(X, y) [X(y==0,:); X(y==1,:)];

%assemble data block based on class to ensure balanced classes
X = block(X, y);
y = block(y, y);

%initialize figure
figure;

%initialize test and mean validation errors
Err_test = zeros(K,1);
Err_validation = zeros(K,1);

%iterate over outer folds
for i = 1:K

    %extract training+validation samples
    X_inner = X(outer ~= i,:);
    y_inner = y(outer ~= i,:);
    
    %re-block data for training/validation
    X_inner = block(X_inner, y_inner);
    y_inner = block(y_inner, y_inner);
    
    %generate inner indices
    inner = mod(1:length(y_inner), K-1)+1;
    
    %allocate error array
    Err = zeros(K-1, length(lambdas));
    
    %iterate over inner folds
    for j = 1:K-1
    
        %start timer
        tic
        
        %generate logistic regression model
        [beta, info] = lassoglm(X_inner(inner ~= j, :), y_inner(inner ~= j),...
            'binomial', 'Alpha', alpha, 'Lambda', lambdas);

        %calculate error
        for b = 1:length(lambdas)
        	y_hat = glmval([beta(:,b); 1], X_inner(inner == j, :),...
                    'logit') >= 0.5;
            Err(j,b) = sum(y_hat ~= y_inner(inner == j)) / length(y_hat);
        end
                
        %update console
        [i j toc]
    
    end
    
    %calculate mean and standard deviation of error
    mu_E = mean(Err, 1);
    sigma_E = std(Err, [], 1);
    
    %identify lambda_min, lambda_star
    [E_min, min_index] = min(mu_E);
    lambda_min = lambdas(min_index);
    candidates = find(mu_E <= mu_E(min_index) + sigma_E(min_index));
    [lambda_star, index] = max(lambdas(candidates));
    star_index = candidates(index);
    
    %capture validation error
    Err_validation(i) = mu_E(star_index);
    
    %generate model with selected parameters and calculate test error
    [beta_star, info] = lassoglm(X(outer ~= i, :), y(outer ~= i),...
            'binomial', 'Alpha', alpha, 'Lambda', lambda_star);
    y_hat = glmval([beta_star; 1], X(outer == i, :), 'logit') >= 0.5;
    Err_test(i) = sum(y_hat ~= y(outer == i)) / length(y_hat);
    
    %generate figure
    subplot(5, 1, i);
    plot(log10(lambdas), mean(Err,1), 'k'); hold on;
    plot(log10(lambdas), mu_E + sigma_E, 'r');   
    plot(log10(lambdas(min_index)), mu_E(min_index), 'bo',...
        'MarkerSize', 10);
    plot(log10(lambdas(star_index)), mu_E(star_index), 'go',...
        'MarkerSize', 10);
    plot(log10(lambdas), mu_E - sigma_E, 'r');
    plot(log10(lambdas), (mu_E(min_index) + sigma_E(min_index)) * ...
        ones(size(lambdas)), 'b--');
    plot(log10(lambdas), (mu_E(min_index) - sigma_E(min_index)) * ...
        ones(size(lambdas)), 'b--');
    plot(log10([lambdas(min_index) lambdas(min_index)]), ylim, 'b--');
    plot(log10([lambdas(star_index) lambdas(star_index)]), ylim, 'g--');    
    ylabel('Classification error');
    if(i == K)
        xlabel('log_{10}(\lambda)');
    end
    if(i == 1)
        legend({'\mu_{Error}', '\mu_{Error} +/- \sigma_{Error}',...
            '\lambda_{min}', '\lambda^*'});
    end
    
end

%plot errors
figure;
boxplot([Err_test, Err_validation], {'Err_{test}', 'Err_{validation}'});
ylabel('Classification error');
