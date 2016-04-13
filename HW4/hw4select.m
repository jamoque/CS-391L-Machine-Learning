function [ newImage, newLabel ] = hw4select( image, label, digit1, digit2 )
% hw4select
% chooses 2 digits, relabel them to -1 and 1, discarding all 8 other digits
% ARGS: 
%     image : MNIST images. image(i)(j) is the jth pixel of the ith image
%     label : label(i) is the digit number of the image(i)
%     digit1: images corresponding to this digit will be relabeled to -1
%     digit2: images corresponding to this digit will be relabeled to 1
% RETURNS:
%     newImage: images containing only digit1 and digit2
%     newLabel: the correspong labels (-1 or 1)

I1 = find (label == digit1);
label (I1) = -1;
I2 = find (label == digit2);
label (I2) = 1;
I = union (I1, I2);
newImage = image (I, :);
newLabel = label (I);
end

