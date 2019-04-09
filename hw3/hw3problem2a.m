load HW3_2.mat
alpha = 0.95;
B = elasticnetlr(X,y,alpha,lambdas);
for i = 1:size(X,2)
    xlim([0.5 2])
    BB = B(i,:);
    plot(log10(lambdas),BB)
    j=length(BB);
    while j>0
        if BB(j) ~= 0
            break
        end
        j=j-1;
    end
    text(log10(lambdas(j+1)),0,""+i)
    hold on
end
xlabel('log_{10}(\lambda)','FontSize',12);
ylabel('\beta','FontSize',12);
hold off

function B = elasticnetlr(x,y,alpha,lambda)
distance = 1e-5;
lambda = sort(lambda(:),1,'descend');
lambda = lambda * size(x,1);
[~,P] = size(x);
B = zeros(P,length(lambda));
b = zeros(P,1);
sqrtvarFun = @(mu) sqrt(mu).*sqrt(1-mu);
devFun = @(mu,y) 2.*(y.*log((y+(y==0))./mu) + (1-y).*log((1-y+(y==1))./(1-mu)));
linkFun = @(mu) log(mu ./ (1-mu));
dlinkFun = @(mu) 1 ./ (mu .* (1-mu));
ilinkFun = @(eta) 1 ./ (1 + exp(-eta));
mu = (y + 0.5) ./ 2;
active = false(1,P);
for i = 1:length(lambda)
    disp(i*100/1000+"%")
    x1 = [ones(size(x,1),1) x];
    b1 = [0;b];
    iter = 0;
    iterLim = 50;
    eta = linkFun(mu);
    while iter <= iterLim
        iter = iter+1;
        deta = dlinkFun(mu);
        sqrtw = 1 ./ (abs(deta) .* sqrtvarFun(mu));
        t = (sqrtw < max(sqrtw)*eps^(2/3));
        if any(t)
            t = t & (sqrtw ~= 0);
            if any(t)
                sqrtw(t) = max(sqrtw)*eps^(2/3);
            end
        end
        Y1 = eta + (y - mu) .* deta;
        weights = sqrtw.^2;
        X0 = x1(:,2:end);
        weights = weights(:)';
        normedWeights = weights / sum(weights);
        muX = normedWeights * X0;
        X0 = bsxfun(@minus,X0,muX);
        muY = normedWeights * Y1;
        Y1 = Y1 - muY;
        bpre = b1(2:end);
        wX = bsxfun(@times,X0,weights');
        wX2 = zeros(P,1);
        wX2(active) = (weights * X0(:,active).^2)';
        wX2calculated = active;
        threshold = lambda(i) * (0.96-alpha);
        shrinkFactor = wX2 + lambda(i) * (alpha);
        while true
            bold = bpre;
            old_active = active;
            r = Y1- X0(:,active)*bpre(active,:);
            bwX2 = bpre.*wX2;
            bold1 = bpre;
            for j=find(active)
                bj = wX(:,j)' * r + bwX2(j);
                margin = abs(bj) - threshold;
                if margin > 0
                    bpre(j) = sign(bj) .* margin ./ shrinkFactor(j);
                else
                    bpre(j) = 0;
                    active(j) = false;
                end
                r = r - X0(:,j)*(bpre(j)-bold1(j));
            end
            if ~any( abs(bpre(old_active) - bold(old_active)) > distance * max(1.0,abs(bold(old_active))) )
                bold = bpre;
                r = Y1 - X0(:,active)*bpre(active,:);
                potentially_active = abs(r' *wX) > threshold;
                new_candidates = potentially_active & ~active;
                if any(new_candidates)
                    r = Y1 - X0(:,active)*bpre(active,:);
                    bold2 = bpre;
                    for j=find(new_candidates)
                        bj = wX(:,j)' * r;
                        margin = abs(bj) - threshold;
                        if margin > 0
                            if ~wX2calculated(j)
                                wX2(j) = weights * X0(:,j).^2;
                                wX2calculated(j) = true;
                                shrinkFactor(j) = wX2(j) + shrinkFactor(j);
                            end
                            bpre(j) = sign(bj) .* margin ./ shrinkFactor(j);
                            active(j) = true;
                        end
                        r = r - X0(:,j)*(bpre(j)-bold2(j));
                    end
                    new_active = active;
                else
                    new_active = active;
                end
                if isequal(new_active, active)
                    break
                else
                    super_active = active | new_active;
                    if ~any( abs(bpre(super_active) - bold(super_active)) > distance * max(1.0,abs(bold(super_active))) )
                        if sum(new_active) > sum(active)
                            bpre = bold;
                        else
                            active = new_active;
                        end
                        break
                    else
                        active = new_active;
                    end
                end
            end
        end
        bpre((range(x)==0),:) = 0;
        Intercept = muY-muX*bpre;
        b1 = [Intercept; bpre];
        eta = x1 * b1;
        mu = ilinkFun(eta);
        if sum(devFun(mu,y)) < distance
            break;
        end
    end
    b1 = b1(2:end);
    B(:,i) = b1;  
    if sum(active) > P
        B = B(:,1:(i-1));
        break
    end 
end 
B = B(:,length(lambda):-1:1);
end