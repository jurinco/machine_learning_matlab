% NeuralNetwork
% by: Nick Slocum
% 
% classdef NeuralNetwork
% end

% X: input features
% X(m,n): input feature n of training sample m
% T: true values for training instances
% T(m,n): truth value for class n for training instance m
% Y: predicted values
% Y(m,n): predicted value for class n for training instance m
% Z{l}(m,n): total input of neuron n in layer l for training instance m
% Y{l}(m,n): total output of neuron n in layer l for training instance m
% 

%% TODO

%% Adaptive learning rate for each connection (6d)

% 
% rprop
% rmsprop
% Yan Lacun's paper "No More Pesky Learning Rates"
% 

% Gradient Descent Methods (typically come in a package)
%   - conjugate gradient
%   - LBFGS
%   - levinberg marcott
%   ...
% 

% the magnitude of gradients varies between layers
%   gradients can get very small in early layers of deep nets
% 
% The fan-in of a unit determines the magnitude of the "overshoot" effects
% caused by simultaneously changing many of the incoming weights of a unit
% to correct the same error.
% 
% 
% 

% Momentum
% Sutskever method (2012)



% Softmax output for classes
exp_z = exp(Z{L});
Y{L} = bsxfun(@rdivide,exp_z,sum(exp_z,2));
% Y{L} = exp_z ./ repmat(sum(exp_z,2),1,3);   % Alternative syntax

dE_dz{L} = Y{L}-T;   % don't need to calc dE_dy{L}

% Use cross-entropy cost function with softmax.
% This will prevent dE_dz from plateauing when y approaches 0 or 1.
C = -(T.*log(Y{L}));   % cross-entropy cost function

% Cross-entropy cost function (for softmax)
xe_cost = -(T.*log(Y{2}));
cost = mean(sum(xe_cost,2),1);

%% Two types of learning algorithms
% Full gradient - graident from all training cases
% optimize smooth non-linear functions
% non-linear conjugate gradient
% Multi-layer NNs are not typical 

% Mini-batch
% prefereable if dataset is large and redundant
% Example Algorithm
%   - guess initial learning rate
%   - if error rate oscillates or continually gets worse -> reduce lr
%   - if error rate falls consistently but slowly -> increase lr
%   - USE VALIDATION SET
%   - towards the end of mini-batch learning it nearly always helps to reduce
%   the lr.  This reduces fluctuations in the weights caused by variations
%   between the gradients of mini-batches.
%   - if the error rate stops decreasing -> reduce lr

%% Tricks
% Random initial weights
% Prevent symmetry between neurons. Symmetry in weights => symmetry in gradients => neurons learn the same feature.
% 
% Smaller incoming weights when the fan-in is big, larger incoming weights
% Prevents learning from overshooting.
% when the fan-in is small.
%   => init_weights should be proportional to sqrt(fan-in)
% scale lr for weights the same way.
% 
% Shift the inputs to have a 0 mean
% Prevents elongation of the error surface.
% 
% Scale the inputs to fit the range (-1, 1)
% Prevents elongation of the error surface.
% 
% Decorrelate the components of the input vector
% - PCA:
%   1. drop components with the smallest eigenvalues
%   2. divide remaining components by the square roots of their eigenvalues
%      => creates a more circular error surface, w/ gradient pointing at the minimum
% 
% Activities of a layer of hidden units should have mean of approx 0.
%   => makes learning faster in the next layer
% 2 * logistic - 1 (hyperbolic tangents between -1 and 1)
% 
% Logistic has the advantage that it ignores large negative inputs
% 


%% Issues
% 1. Optimization: How to use error derivatives on each instance to discover a
% good set of weights?
% 
%   a. How often to update weights?
%       - Online: after each training case
%       - Full batch: after a full sweep through the training cases
%       - Mini-batch: after a random sample of the training cases
%   We might have a large dataset and we don't necessarily need to look at 
%   the entire dataset to fix some weights that we know are quite bad.  We 
%   can instead use a mini-batch of the training set. The batch size can be
%   increased toward the end of training to fine-tune weights.
%  
%   b. How much to update weights?
%       - Use a fixed learning rate. dE/dw * lr
%       - Adapt the global learning rate.
%           * error rate oscillates => reduce lr
%           * error rate steadily decreases => increase lr
%       - Adapt the learning rate on each connection separately.
%       - Don't use steepest descent.
%           When there is an eliptical error landscape, the path of
%           steepest descent can be at almost right angles to the minimum
%           we seek. This situation occurs towards the end of most learning
%           problems.  An alternative to steepest descent might do better.
% 
% 2. Generalization: How to ensure that the learned weights work well for
% cases not seen during training?
% 
%   The training data can contain two types of noise:
%       a. Target values might be unreliable.
%       b Sampling error: there may be accidental regularities (patterns)
%       in the particular cases chosen for training.
% 
%   The model may fit to the noise in the data causing it to overfit the
%   data and preventing it from generalizing well.
% 
%   Reduce overfitting by simplifying the model:
%       - Weight-decay: keep many of the weights small or near zero.
%       - Weight-sharing: multiple weights have the same learned value.
%         e.g. convolutional nets
%       - Early stopping: While training, check the accuracy of the model
%       on a fake test set. Stop training once the performance on the fake
%       test set decreases.
%       - Model averaging: Train many different NNs and average theirp
%       predictions to cancel out errors in individual models.
%       - Bayesian fitting of neural nets: A fancy form of model averaging.
%       - Dropout: make model more robust by randomly ommiting hidden units
%       during training.
%       - Generative pre-training: ?
% 

clear all;

lr = 1;
X = [1 1 0 0;
     1 0 1 0]';
T = [1 0 0 1]';
T = [T ~T];   % add t for 2nd output neuron (use softmax)

m = size(X,1);
n = size(X,2);
nT = size(T,2);

layers = [n 3 nT];
momentum = 0.9;   % viscosity

% Set initial random weights
setdemorandstream(49121812)
W{1} = randn(layers(1),layers(2));
setdemorandstream(79328381)
W{2} = randn(layers(2),layers(3));

target_err = 0.05;
err = 1;
iter = 0;
while err > target_err
    % Feed Forward
    Z{1} = X * W{1};
    Y{1} = 1./(1 + exp(-Z{1}));
    
    Z{2} = Y{1} * W{2};
    Y{2} = 1./(1 + exp(-Z{2}));   % logistic
%     % Softmax 2
%     exp_z = exp(Z{2});
%     Y{2} = bsxfun(@rdivide,exp_z,sum(exp_z,2));
    
    % Backpropogation - Calculate Derivatives
    dE_dy{2} = Y{2} - T;
    dE_dz{2} = Y{2}.*(1-Y{2}) .* dE_dy{2};
%     dE_dz{2} = Y{2}-T;   % softmax (don't need to calc dE_dy{2})
    dE_dw{2} = Y{1}' * dE_dz{2};
    
    dE_dy{1} = dE_dz{2} * W{2}';
%     dE_dy_1 = dE_dz_2 * W{2}';    % mean over all instances (without calculating for each instance)
    dE_dz{1} = Y{1}.*(1-Y{1}) .* dE_dy{1};
    dE_dw{1} = X' * dE_dz{1};
    
    % Update Weights
    W{2} = W{2} - lr * dE_dw{2};
    W{1} = W{1} - lr * dE_dw{1};
    
%     C = 0.5*(T-Y{2});      % Mean error
%     C = 0.5*(T-Y{2}).^2;   % MSE
    C = T.*-log(Y{2});   % cross-entropy cost function (for softmax)
    err = mean(sum(C,2),1);  % sum over all outputs & mean over all instances
    
    iter = iter + 1;
    if (mod(iter,100) == 0) || (iter == 1)
        disp(Y{2});
        disp(err);
        pause(1);
    end
end
disp(iter);
fprintf('error: %.3f\n',err);
% T = 4x1
% logistic, cross-entropy cost
% iter:  784
% error: 0.050
%     0.9388
%     0.0888
%     0.0885
%     0.8722
% softmax, cross-entropy cost


% T = 4x2
% logistic, cross-entropy cost
% iter:  1859
% error: 0.050
%     0.9813    0.0187
%     0.0483    0.9517
%     0.0483    0.9517
%     0.9212    0.0788        
        
% iter:  9767
% error: 0.048
%     0.8526    0.1474
%     0.0165    0.9835
%     0.0165    0.9835
%     1.0000    0.0000
% softmax, cross-entropy cost
% iter:  9767
% error: 0.048
%     0.8526    0.1474
%     0.0165    0.9835
%     0.0165    0.9835
%     1.0000    0.0000

% logistic T:4x1
% iter:  19801
% error: 0.050
% logistic T:4x2
% iter: 33910
% error: 0.050
% softmax T:4x2
% iter: 3932
% error: 0.049

