function [test, train, labels] = hw4Setup(num_training_samples, start_test_index, end_test_index)
    % hw1Setup
        %   PARAMS: 
        %       num_training_samples - number of training samples to use
        %       start_test_index - first test sample to use
        %       end_test_index - last test sample to use
        %   LOADS:
        %       trainImages:    4-D object
        %       trainLabels:    1 x 60,000
        %       testImages:     4-D object
        %       testLabels:     1 x 10,000
        %   RETURNS:
        %       test - matrix of eigen vectors for test images
        %       train - matrix of eigen vectors for training images
        %       labels - training labels for inN images
    
    load digits.mat;
    labels = trainLabels(1:num_training_samples);
    
    % put each training image into a column
    A = [];
    for k = 1 : num_training_samples
        image = trainImages(:,:,1,k);
        A = [A, image(:)];
    end
    
	% call to eigen digits function
    % see hw1FindEigendigits.m
    [m, V] = hw1FindEigendigits(A);
    
    % use only first few eigenvecs
    E = V(:, 1:M)';
    
    % images to eigens - training
    train = [];
    for k = 1 : num_training_samples
        X = trainImages(:,:,1,k);
        X = E * (double(X(:)) - m);       
        train = [train, X];
    end
    train = train';
    
    % images to eigens - testing
    test = [];
    for k = start_test_index : end_test_index
        X = testImages(:,:,1,k);
        X = E * (double(X(:)) - m);        
        test = [test, X];
    end
    test = test';
end