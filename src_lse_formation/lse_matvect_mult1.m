function [JOut_full]=lse_matvect_mult1(JIn0, fN_all, Ae, z_real, z_imag, dx, idx, nodeid_4_grnd, nodeid_4_injectcurr,fl_Tuck,bala)
global fl_precon_type

% -------------------------------------------------------------------------
% Prepare data
% -------------------------------------------------------------------------

fl_volt_source = 2; %symmetric voltage source
fl_gpu = 0;
fl_profile = 0;

tic

num_node=size(Ae,1);
num_curr=size(Ae,2);

% fft dimensions
if fl_Tuck==0
    [LfN, MfN, NfN] = size(fN_all{1});
else
    LfN = fN_all{1}{5}(1); MfN = fN_all{1}{5}(2);  NfN = fN_all{1}{5}(3);
end

% domain dimensions
[L, M, N] = size(z_real);
% L=LfN/2; M=MfN/2; N=NfN/2;
%GPU_flag = 0;

if (fl_gpu == 1)
    % allocate space
    JIn = gpuArray.zeros(L, M, N, 5);    
    JOut = gpuArray.zeros(L, M, N, 5);
    % send to gpu, and translate from local (idx) to global (L,M,N) coordinates
    JIn(idx) = gpuArray(JIn0(1:num_curr));
else

    % allocate space
    JInN = zeros(L, M, N, 5);
    JIn=cell(5,1);
    JIn{1}=zeros(L, M, N);
    JIn{2}=zeros(L, M, N);
    JIn{3}=zeros(L, M, N);
    JIn{4}=zeros(L, M, N);
    JIn{5}=zeros(L, M, N);
    JOut = zeros(L, M, N, 5);
   
    % translate from local (idx) to global (L,M,N) coordinates
    JInN(idx) = JIn0(1:num_curr);
    JIn{1}=JInN(:,:,:,1); JIn{2}=JInN(:,:,:,2); JIn{3}=JInN(:,:,:,3); JIn{4}=JInN(:,:,:,4); JIn{5}=JInN(:,:,:,5);
    clear JInN
end

JOut_full = zeros(num_curr+num_node,1);
JOut_full_in = zeros(num_curr+num_node,1);

%%% allocate space
%JIn = zeros(L, M, N, 3);
%JOut = zeros(L, M, N, 3);
%JOut_full = zeros(num_curr+num_node,1);

% translate from local (idx) to global (L,M,N) coordinates
%JIn(idx) = JIn0(1:num_curr);

% ---------------------------------------------------------------------
% apply fft and mv-op for each of the components of JIn
% ---------------------------------------------------------------------
if isempty(z_imag)
    JOut(:,:,:,1) = (1/dx) .* z_real .* JIn{1};
    JOut(:,:,:,2) = (1/dx) .* z_real .* JIn{2};
    JOut(:,:,:,3) = (1/dx) .* z_real .* JIn{3};
    JOut(:,:,:,4) = (1/(6*dx)) .* z_real .* JIn{4};
    JOut(:,:,:,5) = (1/(2*dx)) .* z_real .* JIn{5};
else
    JOut(:,:,:,1) = (1/dx) .* (z_real+z_imag) .* JIn{1};
    JOut(:,:,:,2) = (1/dx) .* (z_real+z_imag) .* JIn{2};
    JOut(:,:,:,3) = (1/dx) .* (z_real+z_imag) .* JIn{3};
    JOut(:,:,:,4) = (1/(6*dx)) .* (z_real+z_imag) .* JIn{4};
    JOut(:,:,:,5) = (1/(2*dx)) .* (z_real+z_imag) .* JIn{5};
end

JIn{1} = fftn(JIn{1},[LfN, MfN, NfN]);
JIn{2} = fftn(JIn{2},[LfN, MfN, NfN]);
JIn{3} = fftn(JIn{3},[LfN, MfN, NfN]);
JIn{4} = fftn(JIn{4},[LfN, MfN, NfN]);
JIn{5} = fftn(JIn{5},[LfN, MfN, NfN]);
%Jout1
if fl_Tuck==0
    temp = fN_all{1} .* JIn{1}; % Gxx*Jx
    temp = temp + fN_all{2} .* (JIn{4}+JIn{5}); % Gx2d*J2d
else
    temp = bala*ten_mat_prod(fN_all{1}{4},{fN_all{1}{1},fN_all{1}{2},fN_all{1}{3}}) .* JIn{1}; % Gxx*Jx
    temp = temp + bala*ten_mat_prod(fN_all{2}{4},{fN_all{2}{1},fN_all{2}{2},fN_all{2}{3}}) .* (JIn{4}+JIn{5}); % Gx2d*J2d
end
% temp = temp + fN_all{2}(:,:,:) .* JIn{5}; % Gx3d*J3d
temp = ifftn(temp);
if isempty(z_imag)
    JOut(:,:,:,1) = JOut(:,:,:,1) + temp(1:L,1:M,1:N);
else
    JOut(:,:,:,1) = JOut(:,:,:,1) + temp(1:L,1:M,1:N);
end

%Jout2
if fl_Tuck==0
    temp = fN_all{1} .* JIn{2}; % Gyy*Jy
    temp = temp + fN_all{3} .* (JIn{4}-JIn{5}); % Gy2d*J2d - (-(y-yc))
    % temp = temp - fN_all{3}(:,:,:) .* JIn{5}; % Gy3d*J3d - (y-yc)
else
    temp = bala*ten_mat_prod(fN_all{1}{4},{fN_all{1}{1},fN_all{1}{2},fN_all{1}{3}}) .* JIn{2}; % Gyy*Jy
    temp = temp + bala*ten_mat_prod(fN_all{3}{4},{fN_all{3}{1},fN_all{3}{2},fN_all{3}{3}}) .* (JIn{4}-JIn{5}); % Gy2d*J2d - (-(y-yc))
    % temp = temp - fN_all{3}(:,:,:) .* JIn{5}; % Gy3d*J3d - (y-yc)
end
temp = ifftn(temp);
if isempty(z_imag)
    JOut(:,:,:,2) = JOut(:,:,:,2) + temp(1:L,1:M,1:N);
else
    JOut(:,:,:,2) = JOut(:,:,:,2) + temp(1:L,1:M,1:N);
end

%Jout3
if fl_Tuck==0
    temp = fN_all{1} .* JIn{3}; % Gzz*Jz
    temp = temp + fN_all{5} .* JIn{5}; % Gz3d*J3d
else
    temp = bala*ten_mat_prod(fN_all{1}{4},{fN_all{1}{1},fN_all{1}{2},fN_all{1}{3}}) .* JIn{3}; % Gzz*Jz
    temp = temp + bala*ten_mat_prod(fN_all{5}{4},{fN_all{5}{1},fN_all{5}{2},fN_all{5}{3}}) .* JIn{5}; % Gz3d*J3d
end
temp = ifftn(temp);
if isempty(z_imag)
    JOut(:,:,:,3) = JOut(:,:,:,3) + temp(1:L,1:M,1:N);
else
    JOut(:,:,:,3) = JOut(:,:,:,3) + temp(1:L,1:M,1:N);
end

%Jout4
if fl_Tuck==0
    temp = -fN_all{2} .* JIn{1}; % G2dx*Jx
    temp = temp - fN_all{3} .* JIn{2}; % G2dy*Jy - (-(y-yc))
    temp = temp + fN_all{4} .* JIn{4}; % G2d2d*J2d
    temp = temp + fN_all{6} .* JIn{5}; % G2d3d*J3d
else
    temp = -bala*ten_mat_prod(fN_all{2}{4},{fN_all{2}{1},fN_all{2}{2},fN_all{2}{3}}) .* JIn{1}; % G2dx*Jx
    temp = temp - bala*ten_mat_prod(fN_all{3}{4},{fN_all{3}{1},fN_all{3}{2},fN_all{3}{3}}) .* JIn{2}; % G2dy*Jy - (-(y-yc))
    temp = temp + bala*ten_mat_prod(fN_all{4}{4},{fN_all{4}{1},fN_all{4}{2},fN_all{4}{3}}) .* JIn{4}; % G2d2d*J2d
    temp = temp + bala*ten_mat_prod(fN_all{6}{4},{fN_all{6}{1},fN_all{6}{2},fN_all{6}{3}}) .* JIn{5}; % G2d3d*J3d
end
temp = ifftn(temp);
if isempty(z_imag)
    JOut(:,:,:,4) = JOut(:,:,:,4) + temp(1:L,1:M,1:N);
else
    JOut(:,:,:,4) = JOut(:,:,:,4) + temp(1:L,1:M,1:N);
end

%Jout5
if fl_Tuck==0
    temp=-fN_all{2} .* JIn{1}; % G3dx*Jx
    temp = temp + fN_all{3} .* JIn{2}; % G3dy*Jy - (y-yc)
    temp = temp - fN_all{5} .* JIn{3}; % G3dz*Jz
    temp = temp + fN_all{6} .* JIn{4}; % G3d2d*J2d
    temp = temp + fN_all{7} .* JIn{5}; % G3d3d*J3d
else
    temp=-bala*ten_mat_prod(fN_all{2}{4},{fN_all{2}{1},fN_all{2}{2},fN_all{2}{3}}) .* JIn{1}; % G3dx*Jx
    temp = temp + bala*ten_mat_prod(fN_all{3}{4},{fN_all{3}{1},fN_all{3}{2},fN_all{3}{3}}) .* JIn{2}; % G3dy*Jy - (y-yc)
    temp = temp - bala*ten_mat_prod(fN_all{5}{4},{fN_all{5}{1},fN_all{5}{2},fN_all{5}{3}}) .* JIn{3}; % G3dz*Jz
    temp = temp + bala*ten_mat_prod(fN_all{6}{4},{fN_all{6}{1},fN_all{6}{2},fN_all{6}{3}}) .* JIn{4}; % G3d2d*J2d
    temp = temp + bala*ten_mat_prod(fN_all{7}{4},{fN_all{7}{1},fN_all{7}{2},fN_all{7}{3}}) .* JIn{5}; % G3d3d*J3d
end
temp = ifftn(temp);

if isempty(z_imag)
    JOut(:,:,:,5) = JOut(:,:,:,5) + temp(1:L,1:M,1:N);
else
    JOut(:,:,:,5) = JOut(:,:,:,5) + temp(1:L,1:M,1:N);
end

% -------------------------------------------------------------------------
% Return local coordinates related to material positions
% -------------------------------------------------------------------------

if (fl_gpu == 1)
    % get from GPU
    JOut = gather(JOut(idx));
    % clear gpu data
    clear JIn; clear Jout1; clear Jout2; clear Jout3; clear fJ;
else
    JOut = JOut(idx);
end

%JOut = JOut(idx);

JOut_full(1:num_curr) = JOut; 

if(fl_profile == 1); disp(['Time for matvect - fft part::: ',num2str(toc)]); end;

% ---------------------------------------------------------------------
% Adding contributions due to nodal incidence matrix
% ---------------------------------------------------------------------
tic
% Perform multiplications without assigning to dum_block

JOut_full(1:num_curr) = JOut_full(1:num_curr) - (Ae'*JIn0(num_curr+1:num_curr+num_node)) ;

JOut_full(num_curr+1:num_curr+num_node) = Ae*JIn0(1:num_curr);

% this is needed only in case of the original code Schur inversion method, because in this
% case instead of removing the empty rows of Ae, they have been zeroed, so there is the need
% to add dummy equations to the system (see comments in 'lse_sparse_precond_prepare.m'
% relevant to 'DD' matrix)
if ( strcmp(fl_precon_type, 'schur_invert_original') == 1 )
    % For "well-conditioning"
    JOut_full(num_curr+nodeid_4_grnd) = JIn0(num_curr+nodeid_4_grnd);

    if(fl_volt_source == 1 || fl_volt_source == 2)
        JOut_full(num_curr+nodeid_4_injectcurr) = JIn0(num_curr+nodeid_4_injectcurr);
    end
end

if(fl_profile == 1); disp(['Time for matvect - Ae matrices part::: ',num2str(toc)]); end

% ---------------------------------------------------------------------
% Sparse preconditioner [E F; G H]
% ---------------------------------------------------------------------
%tic
%if ( (strcmp(fl_precon_type, 'no_precond') == 0) & (strcmp(fl_precon_type, 'test_no_precond') == 0) )
%    [JOut_full]=lse_sparse_precon_multiply(JOut_full,Ae,nodeid_4_grnd,nodeid_4_injectcurr);
%end
%if(fl_profile == 1); disp(['Time for matvect - sparse preconditioner part::: ',num2str(toc)]); end

% JOut_full_in = JOut_full;
% % block E contribution
% JOut_full(1:num_curr) = A_inv2*JOut_full_in(1:num_curr)+A_inv2*(-Ae')*...
%     QQ * (UU \ (LL \ (PP * (RR \ (Ae*A_inv2*JOut_full_in(1:num_curr))))));
% 
% % block F contribution
% JOut_full(1:num_curr) = JOut_full(1:num_curr)...
%     + A_inv2 * (Ae') * QQ * (UU \ (LL \ (PP * (RR \ (JOut_full_in(num_curr+1:num_curr+num_node))))));
% 
% 
% % block G contribution
% JOut_full(num_curr+1:num_curr+num_node) = ...
%     -QQ * (UU \ (LL \ (PP * (RR \ (Ae*A_inv2*JOut_full_in(1:num_curr))))));
% % block H contribution
% JOut_full(num_curr+1:num_curr+num_node) = JOut_full(num_curr+1:num_curr+num_node)...
%     +QQ * (UU \ (LL \ (PP * (RR \ (JOut_full_in(num_curr+1:num_curr+num_node))))));

% typing one dot for each iteratative solution multiplication step
fprintf ('.') ;

