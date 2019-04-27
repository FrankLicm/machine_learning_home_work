close all; 
clear;
load('HW5.mat')
%data envelope test
d = 2;
k = 3;
n = 1000;
tstdata = toyexample(d,k,n);
tstdata = tstdata';
figure(1)
scatter(tstdata(:,1),tstdata(:,2),".")
hold on
re_data  = generate_reference_data(tstdata);
left_index = find(re_data(:,1) ==min(re_data(:,1)));
right_index = find(re_data(:,1) ==max(re_data(:,1)));
up_index = find(re_data(:,2) ==max(re_data(:,2)));
down_index = find(re_data(:,2) ==min(re_data(:,2)));
x=[min(re_data(:,1)),re_data(up_index(1),1),max(re_data(:,1)),re_data(down_index(1),1),min(re_data(:,1))];
y=[re_data(left_index(1),2),max(re_data(:,2)),re_data(right_index(1),2),min(re_data(:,2)),re_data(left_index(1),2)];
plot(x,y,'-r','lineWidth',1);
legend("data points","data envelope",'Location','northwest')
xlabel('X_1')
ylabel('X_2')
hold off

%gap_statistic
num_clusters = [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ]; 
num_reference_bootstraps = 100;  
[opt_k, gaps, stds] = gap_statistic(X,num_clusters,num_reference_bootstraps);
figure(2)
errorbar(num_clusters,gaps,stds)
hold on
plot(opt_k,gaps(opt_k),"py",'LineWidth',10)
dif_std = abs( gaps - (sqrt(1 + 1/num_reference_bootstraps) * stds));
[xpos, ypos] =  find(( gaps(1:end-1) - dif_std(2:end)) > 0);
for i = 1:length(ypos)
    if ypos(i)==opt_k || ypos(i)< opt_k
        continue
    end
    plot(ypos(i),gaps(ypos(i)),"or",'LineWidth',10)
end
xlabel('K')
ylabel('Gap')
legend("Gap","K^*=16","K|Gap(K)>=Gap(K+1)-s_{k+1}",'Location','northwest')
hold off

function [Z] = generate_reference_data(data)
[~, dim] = size(data); 
[~,~,V] = svd(data);
X_ = data*V;
for i = 1:dim
    maxs(i) = max(X_(:,i));
    mins(i) = min(X_(:,i));
end
for i = 1:length(X_)
    for j = 1:dim
        Z_(i,j) = randi([int16(mins(j)), int16(maxs(j))]);
    end
end
Z = Z_ * V';
end

function [opt_k, gaps, stds] = gap_statistic(data, k_vector, n_tests)
size_k = size(k_vector);
n_done = 0;
dispersions(1, 1:size_k(2)) = zeros;
for id_k = k_vector
    opts = statset('MaxIter', 400);
    [~, ~, sumD] = kmeans(data, id_k, 'EmptyAction', 'singleton', 'options', opts, 'Replicates', 2);
    dispersion = log(sum(sumD));
    dispersions(1, id_k) = dispersion;
    n_done = n_done + 1;
end
for id_test = 2:n_tests+1
    disp(id_test)
    test_data = generate_reference_data(data);
    dispersions(id_test, k_vector) = zeros;
    for id_k = k_vector
            opts = statset('MaxIter', 400);
            [~, ~, sumD] = kmeans(test_data, id_k, 'EmptyAction', 'singleton', 'options', opts, 'Replicates', 2);
            dispersion = log(sum(sumD));
            dispersions(id_test, id_k) = dispersion;
    end
end
gaps(1:size_k(2)) = zeros;
for id_gap = k_vector
    gaps(id_gap) = mean(dispersions(2:n_tests+1,id_gap)) - dispersions(1,id_gap);
end
stds(1:size_k(2)) = zeros;
for id_gap = k_vector
    stds(id_gap) = std(dispersions(2:n_tests+1,id_gap),0);
end
max_gap = max(gaps);
opt_k = find(gaps == max_gap);
b_gaps = gaps ~= 0;
gaps = gaps(b_gaps);
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

