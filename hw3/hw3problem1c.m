load('HW3_1.mat')
warning('off');
disp("please wait for 5-8 minutes")
N = length(X);
k = 5;
index = fix(N/k);
array = [X,y];
shuffledArray = array(randperm(size(array,1)),:);
X=shuffledArray(:,1:34);
y=shuffledArray(:,35);
testerror = zeros(1,k);
outtrainingerror = zeros(length(lambdas),k*(k-1));
outvalidationerror = zeros(length(lambdas),k*(k-1));
mintesterror = inf;
for i=1:k
    XX = X;
    yy = y; 
    if i==k
         X_test = XX(((i-1)*index+1):length(X),:);
         XX(((i-1)*index+1):length(X),:)=[];
         X_train_validate = XX;
         y_test = yy(((i-1)*index+1):length(X));
         yy(((i-1)*index+1):length(X)) = [];
         y_train_validate = yy;
    else
         X_test = XX(((i-1)*index+1):((i-1)*index+index),:);
         XX(((i-1)*index+1):((i-1)*index+index),:)=[];
         X_train_validate = XX;
         y_test = yy(((i-1)*index+1):((i-1)*index+index));
         yy(((i-1)*index+1):((i-1)*index+index)) = [];
         y_train_validate = yy;
    end
    mean_lambda_error = zeros(1,length(lambdas));
    
    standard_lambda_error = zeros(1,length(lambdas));
    for l=1:length(lambdas)
        disp(((i-1)*500+ l)*100/2500+"%")
        N2 = length(X_train_validate);
        k2 = k-1;
        index2 = fix(N2/k2);
        validationerror = zeros(1,k2);
        trainingerror = zeros(1,k2);
        for j=1:k2
            XXX = X_train_validate;
            yyy = y_train_validate;
            if j==k2
                X_validate = XXX(((j-1)*index+1):length(X_train_validate),:);
                XXX(((j-1)*index+1):length(X_train_validate),:)=[];
                X_train = XXX;
                y_validate = yyy(((j-1)*index+1):length(X_train_validate));
                yyy(((j-1)*index+1):length(X_train_validate)) = [];
                y_train = yyy;
            else
                X_validate = XXX(((j-1)*index+1):((j-1)*index+index),:);
                XXX(((j-1)*index+1):((j-1)*index+index),:)=[];
                X_train = XXX;
                y_validate = yyy(((j-1)*index+1):((j-1)*index+index));
                yyy(((j-1)*index+1):((j-1)*index+index)) = [];
                y_train = yyy;
            end
            [B,FitInfo] = lassoglm(X_train,y_train,'binomial',"Alpha",0.95,'Lambda',lambdas(l));
            B0 = FitInfo.Intercept;
            coef = [B0; B];
            yhat_train = glmval(coef,X_train,'logit');
            yhatBinom_train = double(yhat_train>=0.5);
            [c_train,order_train] = confusionmat(y_train,yhatBinom_train);
            if length(c_train) == 2
                trainingerror(j) = 1-(c_train(1,1)+c_train(2,2))/length(y_train);
            else
                trainingerror(j) = 1-(c_train(1,1))/length(y_train);
            end
            outtrainingerror(l,(i-1)*k+j) = trainingerror(j);
            yhat = glmval(coef,X_validate,'logit');
            yhatBinom = double(yhat>=0.5);
            [c,order] = confusionmat(y_validate,yhatBinom);
            if length(c) == 2
                validationerror(j) = 1-(c(1,1)+c(2,2))/length(y_validate);
            else
                validationerror(j) = 1-(c(1,1))/length(y_validate);
            end
            outvalidationerror(l,(i-1)*k+j) = validationerror(j);
        end
        mean_lambda_error(l) = mean(validationerror);
        standard_lambda_error(l) = std(validationerror);
    end
    e1 = mean_lambda_error+standard_lambda_error;
    e2 = mean_lambda_error-standard_lambda_error;
    [argvalue, argmin] = min(mean_lambda_error);
    object = argvalue + standard_lambda_error(argmin);
    for p = argmin:length(lambdas)
        if mean_lambda_error(p)>object
            star = p-1;
            break
        end
    end
    [B,FitInfo] = lassoglm(X_train_validate,y_train_validate,'binomial',"Alpha",0.95,'Lambda',lambdas(star));
    B0 = FitInfo.Intercept;
    coef = [B0; B];
    yhat_test = glmval(coef,X_test,'logit');
    yhatBinom_test = double(yhat_test>=0.5);
    [c_test,order_test] = confusionmat(y_test,yhatBinom_test);
    if length(c_test) == 2
        testerror(i) = 1-(c_test(1,1)+c_test(2,2))/length(y_test);
    else
        testerror(i) = 1-(c_test(1,1))/length(y_test);
    end
    if testerror(i) < mintesterror
        bestlambda_index = star;
    end
end
figure(1)
subplot(2,1,1)
boxplot(testerror','Notch','on','Labels','test error')
subplot(2,1,2)
boxplot([outtrainingerror(bestlambda_index,:)',outvalidationerror(bestlambda_index,:)'],'Notch','on','Labels',{'training error', 'validation error'});
