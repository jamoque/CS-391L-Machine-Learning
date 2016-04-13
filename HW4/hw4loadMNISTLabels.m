function labels = hw4loadMNISTLabels(filename)
% hw4loadMNISTImages
% function:
%     load MNIST labels
% ARGS:
%     filename: the name of the MNIST database file
% RETURNS:
%     labels: a vector of labels

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end
