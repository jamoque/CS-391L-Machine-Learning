function [m, V] = hw1FindEigendigits(A)
    % hw1FindEigendigits
    %   PARAMS: 
    %       matrix A, dimensions x by k
    %           x - total number of pixels in an image and 
    %           k - number of training images
    %   RETURNS:
    %       vector m, length x - "mean vector"
    %       matrix V, dimensions x by k - contains k eigenvectors of cov(A)
    %                                     sorted in descending order of
    %                                     eigenvalue
    
    % average rows
    m = mean(A,2);
    
    A = double(A);
    temp = repmat(m, 1, size(A,2));

    A = A - temp;
    
    % extract eigen vectors and eigen values
    [eigen_vectors, eigen_values] = eig(A' * A);
    
    % sorting
    [eigen_values, idx] = sort(diag(eigen_values), 'descend');
    
    eigen_vectors = A * eigen_vectors;
    eigen_vectors = eigen_vectors(:, idx);
    
    % normalization
    V = normc(eigen_vectors);
    
end