function [e_accuracy, k_accuracy] = hw1Main(num_training_samples, start_test_index, end_test_index)
    % hw1Predict
    %   Serves as main function for all running hw1 algorithms
    %   PARAMS: 
        %       test - matrix of eigen vectors for test images
        %       train - matrix of eigen vectors for training images
        %       labels - labels for training images
        %       num_training_samples - number of training samples to use
        %       start_test_index - first test sample to use
        %       end_test_index - last test sample to use
        %   RETURNS:
        %       k_accuracy - accuracy for k-nearest neighbors approach
        %       e_accuracy - accuracy for experimental approach

    [test, train, labels] = hw1Setup(num_training_samples, start_test_index, end_test_index);
    
    [k_accuracy, e_accuracy] = hw1Predict(test, train, labels, num_training_samples, start_test_index, end_test_index);
    
end