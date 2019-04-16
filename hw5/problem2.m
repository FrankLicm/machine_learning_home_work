close all; 
clear;
load('HW5.mat')
num_clusters = [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ];  
num_reference_bootstraps = 100;  
iter_test = 1;
opt_index = zeros(iter_test,1);
max_gap = zeros(iter_test,0);
compactness_as_dispersion = 0;
while(1)
   [opt_ks,gaps] = gap_statistics(X, num_clusters, num_reference_bootstraps);
   if ~sum(isinf(gaps))
       break;
   end
end

function [opt_ks,gaps] = gap_statistics(data, num_clusters, num_reference_bootstraps)
actual_dispersions = calculate_dispersion(data, num_clusters);
[m_ref_dispersions, std_reference_dispersion] = compute_reference_dispersion(data, num_clusters, num_reference_bootstraps);
gaps = abs(m_ref_dispersions - actual_dispersions);
if(sum(isnan(gaps) |  sum(isinf(gaps))) ||  sum(isnan(std_reference_dispersion) |  sum(isinf(std_reference_dispersion))))
  disp('Warning: There is a NaN or Inf among the results');
  gaps = inf;
  opt_ks = inf;
else
    dif_std = abs( gaps - (sqrt(1 + 1/num_reference_bootstraps) * std_reference_dispersion));
    [xpos,~] =  find((gaps(1:end-1) - dif_std(2:end)) > 0); 
    opt_ks = xpos;
    opt_k = opt_ks(1);
    figure; 
    errorbar(num_clusters, gaps, std_reference_dispersion);
    hold on
    plot(opt_k,gaps(opt_k),"*y",'MarkerSize',10)
    for i = 2:size(opt_ks)
        plot(opt_ks(i), gaps(opt_ks(i)),"or",'MarkerSize',10)
    end
    xlabel('Number of Clusters')
    ylabel('Gaps')
    hold off
end
end

function dispersions = calculate_dispersion(test_data, num_clusters)
dispersions = zeros(length(num_clusters),1); 
[n_samples, ~] = size(test_data);
for kk = 1 : length(num_clusters)
    [ids,~] = kmeans( test_data, num_clusters(kk));
    cluster_mean = mean(test_data);
    for  jj = 1 : kk
        D(jj) = sum(sum(squareform(pdist(test_data(ids == kk ,:))))); 
    end
    if(sum(D)==0)
        for jj = 1:kk
           D(jj) = sum((sum(test_data - repmat(cluster_mean,n_samples,1) ).^2)); 
        end
    end
    dispersions(kk) = log(sum(D));
    if(sum(isinf(dispersions(kk))))
        disp(kk)
        disp(D)
    end
end
end

function [m_reference_dispersion, std_reference_dispersion] = compute_reference_dispersion(reference_data, num_clusters, num_iteration)
         reference_dispersion = zeros( length(num_clusters),  num_iteration);
         for zz = 1 : num_iteration
             generated_data = generate_uniform_points(reference_data);
             reference_dispersion(:,zz) = calculate_dispersion(generated_data, num_clusters);
         end 
         m_reference_dispersion = mean(reference_dispersion,2);
         std_reference_dispersion = std(reference_dispersion, 0, 2);
end

function uniform_points = generate_uniform_points(data_points)       
         [n_points, var_dim] = size(data_points); 
         uniform_points = unifrnd(min(min(data_points)),max(max(data_points)),n_points,var_dim);
end