function [factor_matrix1,factor_matrix2,factor_matrix3,core_tensor,mem,rank] = Tucker(tensor2comp,tolerance)
%TUCKER Summary of this function goes here
%   Detailed explanation goes here

%1) Obtain unfloding matrices
[num_x,num_y,num_z] = size(tensor2comp);

T1 = reshape(tensor2comp,num_x,[]);
T2 = reshape(permute(tensor2comp,[2,1,3]),num_y,[]);
T3 = reshape(permute(tensor2comp,[3,1,2]),num_z,[]);

%2) Compute SVD of unfolding matrices 
[U1, S1, ~] = rsvd(T1,min(size(T1))); %PCA accelerate by randomized scheme
[U2, S2, ~] = rsvd(T2,min(size(T2))); %fast SVD
[U3, S3, ~] = rsvd(T3,min(size(T3)));

%3) Estimate Rank
single_val = abs(diag(S1));
ll_max=size(single_val);
for ll=1:ll_max
    if (single_val(ll)/single_val(1)<tolerance/sqrt(3))
        break    
    else
        continue 
    end    
end
S_rank1=ll;

single_val = abs(diag(S2));
ll_max=size(single_val);
for ll=1:ll_max
    if (single_val(ll)/single_val(1)<tolerance/sqrt(3))
        break    
    else
        continue 
    end    
end
S_rank2=ll;

single_val = abs(diag(S3));
ll_max=size(single_val);
for ll=1:ll_max
    if (single_val(ll)/single_val(1)<tolerance/sqrt(3))
        break    
    else
        continue 
    end    
end
S_rank3=ll;


%4) Tucker Comp
factor_matrix1 = U1(:,1:S_rank1);
factor_matrix2 = U2(:,1:S_rank2);
factor_matrix3 = U3(:,1:S_rank3);
core_tensor = ten_mat_prod(tensor2comp,{factor_matrix1',factor_matrix2',factor_matrix3'});%htucker toolbox

mem = num_x*S_rank1+num_y*S_rank2+num_z*S_rank3+S_rank1*S_rank2*S_rank3;
rank=max([S_rank1,S_rank2,S_rank3]);
end

