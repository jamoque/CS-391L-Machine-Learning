function [W, A, diff] = hw2Main(n, alpha, max)
	load sounds.mat;

	A = rand(5, 5);

	X = A * sounds;

	[W, diff] = hw2ICA(X, n, alpha, max);
end
