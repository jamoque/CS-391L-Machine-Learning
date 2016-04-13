function [] = hw4svmSetup(num_samples)
% SVM with sequential minimal optimization for MNIST
% ARGS: num_samples: the number of training samples to use
% PRINTS: accuracy number for the run

disp('reading data...');
trainingData  = hw4loadMNISTImages ('train-images-idx3-ubyte');
trainingLabel = hw4loadMNISTLabels ('train-labels-idx1-ubyte');

testingData  = hw4loadMNISTImages ('t10k-images-idx3-ubyte');
testingLabel = hw4loadMNISTLabels ('t10k-labels-idx1-ubyte');

disp('initializing matrices...');
[trainingData, trainingLabel] = hw4select (trainingData, trainingLabel, 0, 1);
[testingData , testingLabel ] = hw4select (testingData , testingLabel , 0, 1);

trainingData = trainingData(1:num_samples, :);
trainingLabel = trainingLabel(1:num_samples);
testingData = testingData(1:2000, :);
testingLabel = testingLabel(1:2000);

N = length (trainingLabel);

disp('creating SVM kernel...');

K = zeros (N);

for i = 1:N
    for j = i:N
        K(i, j) = kernel (trainingData(i,:), trainingData(j,:));
    end
end

K = K + triu(K,1)';
K = K + 0.01 * eye(N);

save ('K.mat', 'K')

[C, tolerance, e] = hw4arg2vars (10000, 0.1, 0.1);

[alpha, bias] = hw4svmTrain (K, trainingLabel', C, tolerance);

disp ('making predictions...');

N = length (testingLabel);
M = length (trainingLabel);

correctCount = 0;
correct = 0;

k = zeros (M, 1);
for i = 1:N
    x = testingData (i, :);
    for j = 1:M
        k(j) = kernel (x, trainingData(j, :));
    end
    correct = sum(alpha' .* trainingLabel .* k) + bias > 0;
    correctCount = correctCount + correct;
end

sprintf ('prediction accuracy is: %d', 100 * correctCount / N)
end


function [k] = kernel (X, Y)
    k = 10 * exp(0.1*norm(X-Y)^2/(-2));
end
