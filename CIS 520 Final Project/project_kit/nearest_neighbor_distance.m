function [distance] = nearest_neighbor_distance(inputs)
    %Returns average distance between nearest neighbors
    
    l = size(inputs,1);
    
    distance_matrix = zeros(1, l);
    %Loop through each point
    for i = 1:l
        Y = inputs(i,:);
        X = inputs([1:i-1 i+1:l], :);
        %Get the nearest neighbor in X of Y
        
        knn_idx = knnsearch(X,Y,'K',1);
        
        %Get the actual point
        nearest_neighbor = X(knn_idx,:)
        distance = norm(Y-nearest_neighbor,2);
        distance_matrix(i) = distance;
    
    end
    
    distance = mean(distance_matrix);

end
