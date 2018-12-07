% Returns cell of kernel matrices for a given dataset X
% and cluster centers

function kernel_matrices = get_kernel_matrices(X, centers)
    k = size(centers, 1);
    n = size(X, 1);
    kernel_matrices = NaN(n, k);
    
    % Set kernel width to average distance between nearest neighbors
    ker_width = nearest_neighbor_distance(X);
    for i = 1:k
        kernel = NaN(n, 1);
        for j = 1:n
            kernel(j,1) = exp(-norm(X(j,:) - centers(i,:))^2 / ker_width);
        end
        kernel_matrices(:,i) = kernel;
    end 
end