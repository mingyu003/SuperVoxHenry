function x = dematricize(A, sz, row_dims, col_dims)
%DEMATRICIZE Determine (full) tensor from matricization.
sz(end+1:2) = 1;
d = numel(sz);
row_dims = row_dims(row_dims <= d);
col_dims = col_dims(col_dims <= d);

% Reshape to tensor
x = reshape(A, sz([row_dims, col_dims]) );

% Inverse permute
x = ipermute(x, [row_dims, col_dims]);
