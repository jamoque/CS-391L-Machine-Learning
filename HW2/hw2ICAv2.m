function [M, delta] = hw2ICAv2(A, n, alpha, max)
    % hw2ICAv2
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
		for x = A(:,:)
			B = M * x;
            
			for j = 1:numel(B)
				B(j) = 1 / (1 + exp(-1 * B(j)));
            end
            
            dM = alpha * ((ones(n, 1) - 2 * B) * x' + inv(M'));
			M = M + dM;
		end
		
		delta = [delta; norm(dM)];
	end
end
