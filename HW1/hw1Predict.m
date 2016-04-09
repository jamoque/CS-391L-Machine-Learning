function [k_accuracy, e_accuracy] = hw1Predict(test, train, labels, num_training_samples, start_test_index, end_test_index)
    % hw1Predict
        %   PARAMS: 
        %       test - matrix of eigen vectors for test images
        %       train - matrix of eigen vectors for training images
        %       labels - labels for training images
        %   RETURNS:
        %       k_accuracy - accuracy for k-nearest neighbors approach
        %       e_accuracy - accuracy for experimental approach

    predictions = k_nearest_neighbors(test, train, labels, 15);
    k_accuracy = sum(predictions == testLabels(start_test_index :end_test_index)') / (end_test_index - start_test_index + 1);
    
    predictions = maxLikelihood(testEigens', trainEigens', trainLabels(1:num_training_samples));
    e_accuracy = sum(predictions == testLabels(start_test_index :end_test_index)') / (end_test_index - start_test_index + 1);
    
end