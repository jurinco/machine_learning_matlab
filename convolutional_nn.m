% convolutional_nn

X = [0 1 1 1 0;
     0 0 0 1 0;
     0 1 1 1 0;
     0 0 0 1 0;
     0 1 1 1 0];

W = [1 1 1 0;
     0 0 1 0;
     1 1 1 0;
     0 0 1 0];

% Use Cell Arrays to hold Convolutions
% Conv = {X(1:4,1:4); X(1:4,2:5);
%         X(2:5,1:4); X(2:5,2:5)};
% 
% for i = 1:size(Conv,1)
%     for j = 1:size(Conv,2)
%         Conv_results(i,j) = sum(sum(Conv{i,j}.*W,1),2);
%     end
% end

% Use Matrix with a dim for each Convolution
Conv = zeros(4,4,4);
Conv(:,:,1) = X(1:4,1:4);
Conv(:,:,2) = X(1:4,2:5);
Conv(:,:,3) = X(2:5,1:4);
Conv(:,:,4) = X(2:5,2:5);

nConv = size(Conv,3);
Z = zeros(1,nConv);
for i = 1:nConv
    Z(i) = sum(sum(Conv(:,:,i).*W,1),2);
end

Y = 1./(1 + exp(-Z));   % logistic


%% Pooling

% Maximum
max(Y,2)

% Average
% mean(Y,2)

% Using a linear neuron
Wpool = ones(nConv,1) / nConv;
Y * Wpool;


