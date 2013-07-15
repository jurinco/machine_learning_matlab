% LinearNeuron
% by: Nick Slocum
% 
classdef Neuron
    properties
        bias = 1;             % boolean: include bias term and weight?
        epochs = 50;          % number of epochs for training
        lr = 0.20;            % learning rate for linear regression
        type = 'logistic';    % {'lin_class' 'lin_regress' 'logistic'}
        w;                    % column vector of weights
    end
    
    methods
        function [obj y] = train(obj,X,t)
            m = size(X,1);
            if obj.bias
                X = [ones(m,1) X];
            end
            n = size(X,2);
            
            if isempty(obj.w)
                obj.w = randn(n,1);
            end
            
            curr_epoch = 1;
            nErrors = m;
            while (curr_epoch <= obj.epochs) && (nErrors > 0)
                if strcmp(obj.type,'lin_class')
                    y = obj.predict_lin_class(X);
                    delta_w = obj.lr * sum(X.*repmat(t-y,1,n),1)';
                elseif strcmp(obj.type,'lin_regress')
                    y = obj.predict_lin_regress(X);
                    delta_w = obj.lr * sum(X.*repmat(t-y,1,n),1)';
                elseif strcmp(obj.type,'logistic')
                    y = obj.predict_logistic(X);
                    delta_w = sum(X.*repmat(y.*(1-y).*(t-y),1,n),1)';
                end
                obj.w = obj.w + delta_w;
                
                nErrors = nnz(y~=t);
                mse = sum((t-y).^2) / m;   % mean square error
                
                curr_epoch = curr_epoch + 1;
            end
        end
        
        function y = predict(obj,X)
            m = size(X,1);
            if obj.bias
                X = [ones(m,1) X];
            end
            n = size(X,2);
            
            if strcmp(obj.type,'lin_class')
                y = obj.predict_lin_class(X);
            elseif strcmp(obj.type,'lin_regress')
                y = obj.predict_lin_regress(X);
            elseif strcmp(obj.type,'logistic')
                y = obj.predict_logistic(X);
            end
        end
        
        function y = predict_logistic(obj,X)
            z = X * obj.w;
            y = 1./(1 + exp(-z));
        end
        
        function y = predict_lin_regress(obj,X)
            y = X * obj.w;
        end
        
        function y = predict_lin_class(obj,X)
            z = X * obj.w;
            y = z >= 0;
        end
    end
end

%% Test Cases
% clear all;
% X = [0 0 1 1;
%      0 1 0 1]';
% t = [0 1 1 1]';
% net = Neuron;
% net.type = 'logistic';
% [net y] = net.train(X,t);
% y
% net.predict(X)
% 
% clear all;
% X = [0 0 1 1;
%      0 1 0 1]';
% t = [0 1 1 1]';
% net = Neuron;
% net.type = 'lin_class';
% [net y] = net.train(X,t);
% y
% net.predict(X)
% 
% clear all;
% X = [0 0 1 1;
%      0 1 0 1]';
% t = [0 1 1 1]';
% net = Neuron;
% net.type = 'lin_regress';
% [net y] = net.train(X,t);
% y
% net.predict(X)
