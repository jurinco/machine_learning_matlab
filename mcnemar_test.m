% McNemar test
% 
% INPUTS:
%   t1_n: Set 1 negative examples
%   t1_p: Set 1 positive examples
%   t2_n: Set 2 negative examples
%   t2_p: Set 2 positive examples
% 
function [chi2] = mcnemar_test(t1_n,t2_n)
    b = size(setdiff(t2_n,t1_n),2);
    c = size(setdiff(t1_n,t2_n),2);
    chi2 = (b-c)^2 / (b+c);
end

% function [chi2] = mcnemar_test(t1_p,t1_n,t2_p,t2_n)
%     Table = [size(intersect(t1_n,t2_n),2) size(setdiff(t2_n,t1_n),2);
%              size(setdiff(t1_n,t2_n),2)   size(intersect(t2_p,t1_p),2)];
%     
%     chi2 = (Table{1,2} - Table{2,1})^2 / (Table{1,2} + Table{2,1});
%     
%     cnames = {'Test 2 pos','Test 2 neg'};
%     rnames = {'Test 1 neg','Test 1 neg'};
%     f = figure('Position',[200 200 400 150]);
%     t = uitable('Parent',f,'Data',Table,'ColumnName',cnames,...
%             'RowName',rnames,'Position',[20 20 360 100],'FontSize',16);
% end

%% Test Data
% t2_p = [11:100];
% t2_n = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15];
% t1_p = [2 4 5 7 8 10 11:100];
% t1_n = [1 3 6 9 21];
