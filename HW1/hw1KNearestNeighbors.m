function [predictions] = hw1KNearestNeighbors(test, train, labels, k)
    % hw1KNearestNeighbors
        %   PARAMS: 
        %       test - matrix of eigen vectors for test images
        %       train - matrix of eigen vectors for training images
        %       labels - labels for training images
        %       k - humber of neighbors to consider
        %   RETURNS:
        %       predictions - list of predicted values
    
	predictions = [];

	for i = 1:size(test, 1)
		t = test(i, :);
        
        % sum of squares to find distance
        d = train - repmat(t, size(train, 1), 1);
		d = sum(d .^ 2, 2);
        
        % sort
		[d, order] = sort(d, 'ascend');

        % k-nearest labels
		neighbors = labels(order);
		neighbors = double(neighbors(1:k));
		nearest = mode(neighbors);

		predictions = [predictions; nearest];
	end
end