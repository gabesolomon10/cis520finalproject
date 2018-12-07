% Compute raidal basis func

function beta = rbf(Z, Y)
    Z = [ones(size(Z, 1), 1), Z];
    
    % beta = inv(Z'*Z)(Z'Y)
    beta = inv(Z'*Z)*(Z'*Y);

end
% Y = sum(beta' * Z(i,:))
