function x = ten_mat_prod(x, A)
%TTM N-mode multiplication of (full) tensor with matrix
%
%   Y = TTM(X, A) for a cell array A subsequently computes the N-mode
%   products with the matrices contained in A:
%     Y = ... A{2} o_DIMS(2) A{1} o_DIMS(1) X
%   For an empty entry in A, nothing is done in the corresponding dimension.
%
%   y = ttm(x, {A,B,C,D})     %<-- 4-way multiplication


% Cell A.
%A = {A};
dims = 1:numel(A);
d = max(ndims(x), numel(A));

% Subsequently compute N-mode matrix products A{1} o_(dims{1}),
% A{2} o_(dims{2}), ...

% Calculate size (add singleton dimensions if A contains more entries)
sz = size(x);
sz(end+1:d) = 1;

% Loop over dimensions
for ii=1:length(dims)

  % Ignore empty entries.
  if( isempty(A{ii}) )
    continue;
  end
  
  % Matricize tensor x.
  compl_dims = [1:dims(ii)-1, dims(ii)+1:d];
  
  if(dims(ii) == 1)
    X = matricize(x, dims(ii), compl_dims);
    transX = false;
  else
    X = matricize(x, compl_dims, dims(ii));
    transX = true;
  end
  
  % Apply matrix A{ii}
  if(isfloat(A{ii}))
    if(size(A{ii}, 2) ~= sz(dims(ii)))
        error('Matrix dimensions must agree.')
    end
    if(~transX)
        X = A{ii}*X;
    else
        X = X*A{ii}.';
    end
  end
  
  % Update size of X
  if(transX == false)
    sz(dims(ii)) = size(X, 1);
  else
    sz(dims(ii)) = size(X, 2);
  end

  % Dematricize X
  if(transX == false)
    x = dematricize(X, sz, dims(ii), compl_dims);
  else
    x = dematricize(X, sz, compl_dims, dims(ii));
  end
end
