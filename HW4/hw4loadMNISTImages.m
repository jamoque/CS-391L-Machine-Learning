function images = hw4loadMNISTImages(filename)
% hw4loadMNISTImages
% function:
%     load MNIST images
% ARGS:
%     filename: the name of the MNIST database file
% RETURNS:
%     image: A vector of pixel image.
%            image(i) is the ith image
%            image(i)(j) is the jth pixel of the ith image

fp = fopen(filename, 'rb');

assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows   = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols   = fread(fp, 1, 'int32', 0, 'ieee-be');

images    = fread(fp, inf, 'unsigned char');

images    = reshape(images, numCols * numRows, numImages);

images    = double(images)' / 255;
fclose(fp);
end
