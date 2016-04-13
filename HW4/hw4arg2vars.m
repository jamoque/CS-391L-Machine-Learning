function varargout = hw4arg2vars(varargin)
% hw4arg2vars   
% Converts input arguments to variables

nj = 0;
no = nargout;
varargout = cell(1,no);                     

for j = 1:nargin
    if j>no, break, end
    vj = varargin{j};
    if iscell(vj) && ~iscell(vj{1})
        x = cell2mat(vj);
        n = min(no-nj,length(x));
        for k = 1:n
            varargout(nj+k) = {x(k)};
        end
    else
        n = 1;
        varargout(nj+1) = {vj};
    end
    nj = nj+n;
end