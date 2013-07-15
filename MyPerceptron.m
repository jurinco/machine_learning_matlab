% MyPerceptron
% by: Nick Slocum
% 
% Perceptron: online linear classifier
% 
classdef MyPerceptron
properties
    weights;
    epochs = 100;
end

methods
    function [obj] = train(obj,X,t)
        m = size(X,1);
        n = size(X,2);
        
        X = [ones(m,1) X];   % Add bias term
        
        curr_epoch = 1;
        nErrors = m;
        while (curr_epoch < obj.epochs) && (nErrors > 0)
            nErrors = 0;
            
            % Iterate through each training case
            for i = 1:m
                z = X(i,:) * obj.weights;
                y = z >= 0;   % y=1 if z>=0 else y=0;
                
                if y ~= t(i)
                    nErrors = nErrors + 1;
                    if y == 0
                        obj.weights = obj.weights + X(i,:)';
                    else % y == 1
                        obj.weights = obj.weights - X(i,:)';
                    end
                end
            end
            fprintf('Epoch: %d   Errors: %d\n',curr_epoch,nErrors);
            curr_epoch = curr_epoch + 1;
        end
    end
    
    function [y] = predict(obj,X)
        m = size(X,1);
        z = [ones(m,1) X] * obj.weights;
        y = z >= 0;
    end
end
end


% % Test Cases
% X = [0 0 1 1; 0 1 0 1]';
% t = [0 1 1 1]';
% 
% X = [0 0 0 0 1 1 1 1;
%      0 0 1 1 0 0 1 1;
%      0 1 0 1 0 1 0 1]';
% t = [0 1 1 1 1 1 1 1]';
% 
% 
% X = [0 0 1 1;
%      0 1 0 1]';
% t = [1 0 0 1]';
% 
% 
% X = [0 0 1 1; 0 1 0 1];
% t = [0 1 1 1];
% 
% net = CustomPerceptron;
% net = net.train(X,t);
% net.weights
% y = net.predict(X)


% % Equivalent using Matlab functions
% net = perceptron;
% net = train(net,X,t);
% y = net(X);
