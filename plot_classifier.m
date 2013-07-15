% plot_classifier
% by: Nick Slocum
% 
function plot_classifier(X,t,y,w)
    % X: matrix of features
    % t: columm vector of true classes
    % y: column vector of predicted classes
    % w: weights for each feature, w(1) is the bias term
    % func: classifier boundary function
    markers = ['o' 's' 'x' '^' 'd'];
    groups = unique(t);
    
    figure;
    hold on;
    for i = 1:length(groups)
        idx = find(t==groups(i));
        true_idx = intersect(idx,find(y == t));   % Correctly classified
        false_idx = setdiff(idx,true_idx);        % Inorrectly classified
        plot(X(true_idx,1),X(true_idx,2),[markers(i) 'g'], 'markersize',20);
        plot(X(false_idx,1),X(false_idx,2),[markers(i) 'r'], 'markersize',20);
    end
    
    plot([-1,1],[(-w(1)+1*w(2))/w(3),(-w(1)-1*w(2))/w(3)],'k')
    xlim([-0.2,1.2]);
    ylim([-0.2,1.2]);
    hold off;
    
%     % Set axes
%     % xmin xmax
%     min(X(:,1))
%     max(X(:,1))
%     % ymin ymax
%     min(X(:,2))
%     max(X(:,2))
    
end
