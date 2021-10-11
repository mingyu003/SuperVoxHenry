function A = matricize(x, row_dims, col_dims)
%MATRICIZE Matricization of (full) tensor.

row_dims = row_dims(row_dims <= ndims(x));
col_dims = col_dims(col_dims <= ndims(x));  

% Save size of original tensor
sz = size(x);

% Permute dimensions of tensor x
x = permute(x, [row_dims, col_dims]);

% Reshape to matrix
A = reshape(x, prod(sz(row_dims)), prod(sz(col_dims)) );
