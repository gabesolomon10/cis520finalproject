% Compute nearest cluster center for K means as well as SSE
function [clusters, tot_error] = get_closest_cluster(X, C)
   % For each X(i,:), compute closest cluster center
   n = size(X, 1);
   K = size(C, 1);
   clusters = NaN(n, 1);
   tot_error = 0;
   for i = 1:n
       % Compute L2 distance to each cluster center
       dists = sum((repmat(X(i,:), K, 1) - C).^2, 2);
       for j = 1:K
           dists(j,:) = sqrt(dists(j,:));
       end
       % Find cluster assignment & increment SSE
       [m,idx] = min(dists);
       clusters(i) = C(idx);
       tot_error = tot_error + m;
   end
end