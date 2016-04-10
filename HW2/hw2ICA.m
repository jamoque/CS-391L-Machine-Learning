function [M, delta] = hw2ICA(A, n, alpha, max)
    % hw2ICA
    %   PARAMS: 
    %       A - mixed sound signals
    %       n - used for populating random matrix
    %       alpha - learning rate
    %       max - maximum number of iterations
    %   RETURNS:
    %       M - matrix that recovers the original n source signals
    %       delta - changes from gradient descent
    
    delta = [];
    
	% intially populate M with small random numbers
	M = rand(n, size(A, 1)) / 10.0;
    
    % update M using gradient descent
	for i = 0:max
		B = M * A;
        
        for j = 1:numel(B)
			B(j) = 1.0 / (1.0 + exp(-1.0 * B(j)));
        end
        
        C = size(A, 2) * eye(n);
        D = (ones(n, size(A, 2)) - 2 * B) * (M * A)';
		dM = alpha * (C + D) * M;
		M = M + dM;

		if mod(i, 10) == 0
			delta = [delta; norm(dM)];
		end
	end
end
