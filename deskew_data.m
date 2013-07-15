%% deskew_data
% 
% return the original data if the original was not deskewed
% return -1 if the original image was not deskewed
% 

function [dData, mod_angle] = deskew_data(Data, mod_function, min, max)

% Compute the PC (principal componant)
Covar = cov(Data);
[Eigenvectors Eigenvalues] = eig(Covar);

eigenvalues = diag(Eigenvalues);
[eigenvalues perm] = sort(eigenvalues,'descend');   % Sort eigenvalues and get the sorting permutation.
Eigenvectors = Eigenvectors(:,perm);    % Sort eigenvectors


% Calculate the skew of the image.
% The skew is the angle between the PC's eigenvector and the y-axis
angle = atand(Eigenvectors(1,1)/Eigenvectors(2,1));

% Calculate how much of the variance is explained by each variable
var =  eigenvalues ./ sum(eigenvalues);


% % Plot data with eigenvector
% Data0 = bsxfun(@minus, Data, mean(Data));
% 
% figure
% plot(Data0(:,1), Data0(:,2), 'b.');
% hold on;
% plot([0 Eigenvectors(1,2)], [0 Eigenvectors(2,2)], 'r-');   % first eigenvector
% plot([0 Eigenvectors(1,1)], [0 Eigenvectors(2,1)], 'g-');   % second eigenvector
% axis equal


% Modulate the skew angle according to how much variance is explained by the PC.
mod_angle = angle * var(1)^(((1-var(1))*10)^2);

if abs(mod_angle) >= 1
    % Deskew the image
    dData = imrotate(Data, mod_angle, 'bilinear', 'crop');
else
    dData = Data;
end
