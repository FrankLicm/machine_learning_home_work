load('HW3_1.mat')
warning('off');
disp("please wait for 5-8 minutes")
N = length(X);
k = 5;
array = [X,y];
shuffledArray = array(randperm(size(array,1)),:);
X=shuffledArray(:,1:34);
y=shuffledArray(:,35);
index = fix(N/k);
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
            yhat = glmval(coef,X_validate,'logit');
            yhatBinom = double(yhat>=0.5);
            [c,order] = confusionmat(y_validate,yhatBinom);
            if length(c) == 2
                validationerror(j) = 1-(c(1,1)+c(2,2))/length(y_validate);
            else
                validationerror(j) = 1-(c(1,1))/length(y_validate);
            end
        end
        mean_lambda_error(l) = mean(validationerror);
        standard_lambda_error(l) = std(validationerror);
    end
    e1 = mean_lambda_error+standard_lambda_error;
    e2 = mean_lambda_error-standard_lambda_error;
    [argvalue, argmin] = min(mean_lambda_error);
    object = argvalue + standard_lambda_error(argmin);
    star = length(lambdas);
    while mean_lambda_error(star)>object
        star=star-1;
    end
    lambda = log10(lambdas);
    figure(1)
    subplot(5,1,i)
    plot(lambda,e1,"r-")
    hold on
    plot(lambda,mean_lambda_error,"k-")
    plot(lambda,e2,"r-")
    plot(lambda(argmin),argvalue,"bo")
    X11 = -5:0.1:2;
    Y33 = (argvalue + standard_lambda_error(argmin))*ones(size(X11));
    Y44 = (argvalue - standard_lambda_error(argmin))*ones(size(X11));
    plot(X11, Y33,"b--");
    plot(X11, Y44,"b--");
    Y11 = 0:0.01:0.4;
    X33 = lambda(argmin) * ones(size(Y11));
    plot(X33, Y11,"b--");
    X44 = lambda(star) * ones(size(Y11));
    plot(X44, Y11,"g--");
    plot(lambda(star),mean_lambda_error(star),"go")
    %legend("\mu_{Error}+\sigma_{Error}","\mu_{Error}","\mu_{Error}-\sigma_{Error}","\lambda_{min}","\lambda_{star}",'Location','northwest')
    xlabel('log_{10}(\lambda)','FontSize',12);
    ylabel('Classification error','FontSize',12); 
end
set(gcf, 'Position', get(0, 'Screensize'));
hold off


