function [predictions] = hw1MaxLikelihood(test, train, labels)
	% hw1MaxLikelihood
        %   PARAMS: 
        %       test - matrix of eigen vectors for test images
        %       train - matrix of eigen vectors for training images
        %       labels - labels for training images
        %   RETURNS:
        %       predictions - list of predicted values
    
    predictions = [];
	m = [];
	c = [];
    digits = (0:9)';
    
    % build up means and covariances 
	for i = 1:size(digits)
		imgs = train((labels == (i - 1)), :);
        c(:, :, i) = c(imgs);
		m(:, i) = mean(imgs, 1)';
    end
    
    % make prediction
	for t = test(:, :)'
		p = [];
		for l = 1:size(digits)
			p = [p, mvnpdf(t, m(:, l), c(:, :, l))];
		end

		[v, i] = max(p);
		predictions = [predictions; digits(i)];
	end
end
