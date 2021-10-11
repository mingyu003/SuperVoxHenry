function [fN_all_comp,st_sparse_precon] = retrieval_circulant1(dx,ko,L,M,N,fl_no_fft,pre_data_far_med,fl_Tucker,tol_Tucker)
tic
fl_optim_FFT=1;
LMN_new=FindOptimumFFTDims([L M N]);
diffx=2*(LMN_new(1)-L); diffy=2*(LMN_new(2)-M); diffz=2*(LMN_new(3)-N);
if fl_optim_FFT==0
    LMN_new(1)=L; LMN_new(2)=M; LMN_new(3)=N;
    diffx=0; diffy=0; diffz=0;
end
if fl_Tucker==1
    fN_all_comp=cell(7,1);
end
    
% G_mn = zeros(L,M,N,7);
G_mn = cell(7,1);
fN_all= cell(7,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cube 'L'
[Gp_coeff_L] = gperiodic_coeff_nop_linJVIE('L');
% Cube 'M'
[Gp_coeff_M] = gperiodic_coeff_nop_linJVIE('M');
% Cube 'N'
[Gp_coeff_N] = gperiodic_coeff_nop_linJVIE('N');
% Cube 'LM'
[Gp_coeff_LM] = gperiodic_coeff_nop_linJVIE('LM');
% Cube 'LN'
[Gp_coeff_LN] = gperiodic_coeff_nop_linJVIE('LN');
% Cube 'MN'
[Gp_coeff_MN] = gperiodic_coeff_nop_linJVIE('MN');
% Cube 'LMN'
[Gp_coeff_LMN] = gperiodic_coeff_nop_linJVIE('LMN');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% infofN = whos('G_mn'); memestimated = 2*infofN.bytes/(1024*1024);
% disp(['  Memory for temporarily storing G_mn (MB) ::: ' , num2str(memestimated)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% block 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
temp=ten_mat_prod(pre_data_far_med{4}{1},{pre_data_far_med{1}{1},pre_data_far_med{2}{1},pre_data_far_med{3}{1}});
Tucker_Toeplitz=7*toc
G_mn{1}(1:L,1:M,1:N)=dx.*temp(1:L,1:M,1:N); G_mn{1}=G_mn{1}* ko^2;
temp_tens=zeros(2*LMN_new(1),2*LMN_new(2),LMN_new(3));
temp_tens(1:L,1:M,1:N) = G_mn{1};
% Gp_mn{1}(1:L,1:M,1:N) = G_mn{1};
% Cube 'L'
temp_tens(L+2+diffx:2*L+diffx,1:M,1:N)                                   = G_mn{1}(L:-1:2,1:M,1:N) * Gp_coeff_L(1,1);
% Gp_mn{1}(L+2:2*L,1:M,1:N)         = G_mn{1}(L:-1:2,1:M,1:N) * Gp_coeff_L(1,1);
% Cube 'M'
temp_tens(1:L,M+2+diffy:2*M+diffy,1:N)                                   = G_mn{1}(1:L,M:-1:2,1:N) * Gp_coeff_M(1,1);
% Gp_mn{1}(1:L,M+2:2*M,1:N)         = G_mn{1}(1:L,M:-1:2,1:N) * Gp_coeff_M(1,1);
% Cube 'N'
temp_tens(1:L,1:M,N+2+diffz:2*N+diffz)                                   = G_mn{1}(1:L,1:M,N:-1:2) * Gp_coeff_N(1,1);
% Gp_mn{1}(1:L,1:M,N+2:2*N)         = G_mn{1}(1:L,1:M,N:-1:2) * Gp_coeff_N(1,1);
% Cube 'LM'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,1:N)                   = G_mn{1}(L:-1:2,M:-1:2,1:N) * Gp_coeff_LM(1,1);
% Gp_mn{1}(L+2:2*L,M+2:2*M,1:N)     = G_mn{1}(L:-1:2,M:-1:2,1:N) * Gp_coeff_LM(1,1);
% Cube 'LN'
temp_tens(L+2+diffx:2*L+diffx,1:M,N+2+diffz:2*N+diffz)                   = G_mn{1}(L:-1:2,1:M,N:-1:2) * Gp_coeff_LN(1,1);
% Gp_mn{1}(L+2:2*L,1:M,N+2:2*N)     = G_mn{1}(L:-1:2,1:M,N:-1:2) * Gp_coeff_LN(1,1);
% Cube 'MN'
temp_tens(1:L,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz)                 = G_mn{1}(1:L,M:-1:2,N:-1:2) * Gp_coeff_MN(1,1);
% Gp_mn{1}(1:L,M+2:2*M,N+2:2*N)     = G_mn{1}(1:L,M:-1:2,N:-1:2) * Gp_coeff_MN(1,1);
% Cube 'LMN'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz) = G_mn{1}(L:-1:2,M:-1:2,N:-1:2) * Gp_coeff_LMN(1,1);
% Gp_mn{1}(L+2:2*L,M+2:2*M,N+2:2*N) = G_mn{1}(L:-1:2,M:-1:2,N:-1:2) * Gp_coeff_LMN(1,1);

% Obtain the FFTs of circulant
if (fl_no_fft == 1)
    fN_all{1} = temp_tens;
else
    tic
    fN_all{1} = fft_operator(temp_tens);
    disp(['  Time for FFT of block circulant tensor 1 ::: ',num2str(toc)])  
    if fl_Tucker==1
        %%% Tucker compression and figuring out the computational overhead
        fei=tic;
        fN_all_comp{1}=cell(5,1);
        [fact_mat1,fact_mat2,fact_mat3,core_tens,mem_tens1,~] = Tucker(fN_all{1},tol_Tucker);
        fN_all_comp{1}{1}=fact_mat1;
        fN_all_comp{1}{2}=fact_mat2;
        fN_all_comp{1}{3}=fact_mat3;
        fN_all_comp{1}{4}=core_tens;
        fN_all_comp{1}{5}=size(fN_all{1});
        fN_all{1}=[];
        aifei=toc(fei);
        disp(['  Time for Tucker compression block 1::: ',num2str(aifei)])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% block 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp=ten_mat_prod(pre_data_far_med{4}{2},{pre_data_far_med{1}{2},pre_data_far_med{2}{2},pre_data_far_med{3}{2}});
G_mn{2}(1:L,1:M,1:N)=(dx).*temp(1:L,1:M,1:N); G_mn{2}=G_mn{2}* ko^2;
temp_tens=zeros(2*LMN_new(1),2*LMN_new(2),LMN_new(3));
temp_tens(1:L,1:M,1:N) = G_mn{2};
% Cube 'L'
temp_tens(L+2+diffx:2*L+diffx,1:M,1:N)                                   = G_mn{2}(L:-1:2,1:M,1:N) * Gp_coeff_L(2,1);
% Cube 'M'
temp_tens(1:L,M+2+diffy:2*M+diffy,1:N)                                   = G_mn{2}(1:L,M:-1:2,1:N) * Gp_coeff_M(2,1);
% Cube 'N'
temp_tens(1:L,1:M,N+2+diffz:2*N+diffz)                                   = G_mn{2}(1:L,1:M,N:-1:2) * Gp_coeff_N(2,1);
% Cube 'LM'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,1:N)                   = G_mn{2}(L:-1:2,M:-1:2,1:N) * Gp_coeff_LM(2,1);
% Cube 'LN'
temp_tens(L+2+diffx:2*L+diffx,1:M,N+2+diffz:2*N+diffz)                   = G_mn{2}(L:-1:2,1:M,N:-1:2) * Gp_coeff_LN(2,1);
% Cube 'MN'
temp_tens(1:L,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz)                 = G_mn{2}(1:L,M:-1:2,N:-1:2) * Gp_coeff_MN(2,1);
% Cube 'LMN'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz) = G_mn{2}(L:-1:2,M:-1:2,N:-1:2) * Gp_coeff_LMN(2,1);


G_mn{2}=[];
% Obtain the FFTs of circulant
if (fl_no_fft == 1)
    fN_all{2} = temp_tens;
else
    tic
    fN_all{2} = fft_operator(temp_tens);
    disp(['  Time for FFT of block circulant tensor 2 ::: ',num2str(toc)])
    if fl_Tucker==1
        %%% Tucker compression and figuring out the computational overhead
        fei=tic;
        fN_all_comp{2}=cell(5,1);
        [fact_mat1,fact_mat2,fact_mat3,core_tens,mem_tens2,~] = Tucker(fN_all{2},tol_Tucker);
        fN_all_comp{2}{1}=fact_mat1;
        fN_all_comp{2}{2}=fact_mat2;
        fN_all_comp{2}{3}=fact_mat3;
        fN_all_comp{2}{4}=core_tens;
        fN_all_comp{2}{5}=size(fN_all{2});
        fN_all{2}=[];
        aifei=toc(fei);
        disp(['  Time for Tucker compression block 2::: ',num2str(aifei)])
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% block 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

temp=ten_mat_prod(pre_data_far_med{4}{3},{pre_data_far_med{1}{3},pre_data_far_med{2}{3},pre_data_far_med{3}{3}});
G_mn{3}(1:L,1:M,1:N)=(dx).*temp(1:L,1:M,1:N); G_mn{3}=G_mn{3}* ko^2;
temp_tens=zeros(2*LMN_new(1),2*LMN_new(2),LMN_new(3));
temp_tens(1:L,1:M,1:N) = G_mn{3};
% Cube 'L'
temp_tens(L+2+diffx:2*L+diffx,1:M,1:N)                                   = G_mn{3}(L:-1:2,1:M,1:N) * Gp_coeff_L(3,1);
% Cube 'M'
temp_tens(1:L,M+2+diffy:2*M+diffy,1:N)                                   = G_mn{3}(1:L,M:-1:2,1:N) * Gp_coeff_M(3,1);
% Cube 'N'
temp_tens(1:L,1:M,N+2+diffz:2*N+diffz)                                   = G_mn{3}(1:L,1:M,N:-1:2) * Gp_coeff_N(3,1);
% Cube 'LM'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,1:N)                   = G_mn{3}(L:-1:2,M:-1:2,1:N) * Gp_coeff_LM(3,1);
% Cube 'LN'
temp_tens(L+2+diffx:2*L+diffx,1:M,N+2+diffz:2*N+diffz)                   = G_mn{3}(L:-1:2,1:M,N:-1:2) * Gp_coeff_LN(3,1);
% Cube 'MN'
temp_tens(1:L,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz)                 = G_mn{3}(1:L,M:-1:2,N:-1:2) * Gp_coeff_MN(3,1);
% Cube 'LMN'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz) = G_mn{3}(L:-1:2,M:-1:2,N:-1:2) * Gp_coeff_LMN(3,1);


G_mn{3}=[];
% Obtain the FFTs of circulant
if (fl_no_fft == 1)
    fN_all{3} = temp_tens;
else
    tic
    fN_all{3} = fft_operator(temp_tens);
    disp(['  Time for FFT of block circulant tensor 3 ::: ',num2str(toc)])
    if fl_Tucker==1
        %%% Tucker compression and figuring out the computational overhead
        fei=tic;
        fN_all_comp{3}=cell(5,1);
        [fact_mat1,fact_mat2,fact_mat3,core_tens,mem_tens3,~] = Tucker(fN_all{3},tol_Tucker);
        fN_all_comp{3}{1}=fact_mat1;
        fN_all_comp{3}{2}=fact_mat2;
        fN_all_comp{3}{3}=fact_mat3;
        fN_all_comp{3}{4}=core_tens;
        fN_all_comp{3}{5}=size(fN_all{3});
        fN_all{3}=[];
        aifei=toc(fei);
        disp(['  Time for Tucker compression block 3::: ',num2str(aifei)])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% block 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp=ten_mat_prod(pre_data_far_med{4}{4},{pre_data_far_med{1}{4},pre_data_far_med{2}{4},pre_data_far_med{3}{4}});
G_mn{4}(1:L,1:M,1:N)=(dx).*temp(1:L,1:M,1:N); G_mn{4}=G_mn{4}* ko^2;
temp_tens=zeros(2*LMN_new(1),2*LMN_new(2),LMN_new(3));
temp_tens(1:L,1:M,1:N) = G_mn{4};
% Cube 'L'
temp_tens(L+2+diffx:2*L+diffx,1:M,1:N)                                   = G_mn{4}(L:-1:2,1:M,1:N) * Gp_coeff_L(4,1);
% Cube 'M'
temp_tens(1:L,M+2+diffy:2*M+diffy,1:N)                                   = G_mn{4}(1:L,M:-1:2,1:N) * Gp_coeff_M(4,1);
% Cube 'N'
temp_tens(1:L,1:M,N+2+diffz:2*N+diffz)                                   = G_mn{4}(1:L,1:M,N:-1:2) * Gp_coeff_N(4,1);
% Cube 'LM'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,1:N)                   = G_mn{4}(L:-1:2,M:-1:2,1:N) * Gp_coeff_LM(4,1);
% Cube 'LN'
temp_tens(L+2+diffx:2*L+diffx,1:M,N+2+diffz:2*N+diffz)                   = G_mn{4}(L:-1:2,1:M,N:-1:2) * Gp_coeff_LN(4,1);
% Cube 'MN'
temp_tens(1:L,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz)                 = G_mn{4}(1:L,M:-1:2,N:-1:2) * Gp_coeff_MN(4,1);
% Cube 'LMN'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz) = G_mn{4}(L:-1:2,M:-1:2,N:-1:2) * Gp_coeff_LMN(4,1);


% Obtain the FFTs of circulant
if (fl_no_fft == 1)
    fN_all{4} = temp_tens;
else
    tic
    fN_all{4} = fft_operator(temp_tens);
    disp(['  Time for FFT of block circulant tensor 4 ::: ',num2str(toc)])
    if fl_Tucker==1
        %%% Tucker compression and figuring out the computational overhead
        fei=tic;
        fN_all_comp{4}=cell(5,1);
        [fact_mat1,fact_mat2,fact_mat3,core_tens,mem_tens4,~] = Tucker(fN_all{4},tol_Tucker);
        fN_all_comp{4}{1}=fact_mat1;
        fN_all_comp{4}{2}=fact_mat2;
        fN_all_comp{4}{3}=fact_mat3;
        fN_all_comp{4}{4}=core_tens;
        fN_all_comp{4}{5}=size(fN_all{4});
        fN_all{4}=[];
        aifei=toc(fei);
        disp(['  Time for Tucker compression block 4::: ',num2str(aifei)])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% block 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp=ten_mat_prod(pre_data_far_med{4}{5},{pre_data_far_med{1}{5},pre_data_far_med{2}{5},pre_data_far_med{3}{5}});
G_mn{5}(1:L,1:M,1:N)=(dx).*temp(1:L,1:M,1:N); G_mn{5}=G_mn{5}* ko^2;
temp_tens=zeros(2*LMN_new(1),2*LMN_new(2),LMN_new(3));
temp_tens(1:L,1:M,1:N) = G_mn{5};
% Cube 'L'
temp_tens(L+2+diffx:2*L+diffx,1:M,1:N)                                   = G_mn{5}(L:-1:2,1:M,1:N) * Gp_coeff_L(5,1);
% Cube 'M'
temp_tens(1:L,M+2+diffy:2*M+diffy,1:N)                                   = G_mn{5}(1:L,M:-1:2,1:N) * Gp_coeff_M(5,1);
% Cube 'N'
temp_tens(1:L,1:M,N+2+diffz:2*N+diffz)                                   = G_mn{5}(1:L,1:M,N:-1:2) * Gp_coeff_N(5,1);
% Cube 'LM'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,1:N)                   = G_mn{5}(L:-1:2,M:-1:2,1:N) * Gp_coeff_LM(5,1);
% Cube 'LN'
temp_tens(L+2+diffx:2*L+diffx,1:M,N+2+diffz:2*N+diffz)                   = G_mn{5}(L:-1:2,1:M,N:-1:2) * Gp_coeff_LN(5,1);
% Cube 'MN'
temp_tens(1:L,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz)                 = G_mn{5}(1:L,M:-1:2,N:-1:2) * Gp_coeff_MN(5,1);
% Cube 'LMN'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz) = G_mn{5}(L:-1:2,M:-1:2,N:-1:2) * Gp_coeff_LMN(5,1);

G_mn{5}=[];
% Obtain the FFTs of circulant
if (fl_no_fft == 1)
    fN_all{5} = temp_tens;
else
    tic
    fN_all{5} = fft_operator(temp_tens);
    disp(['  Time for FFT of block circulant tensor 5 ::: ',num2str(toc)])
    if fl_Tucker==1
        %%% Tucker compression and figuring out the computational overhead
        fei=tic;
        fN_all_comp{5}=cell(5,1);
        [fact_mat1,fact_mat2,fact_mat3,core_tens,mem_tens5,~] = Tucker(fN_all{5},tol_Tucker);
        fN_all_comp{5}{1}=fact_mat1;
        fN_all_comp{5}{2}=fact_mat2;
        fN_all_comp{5}{3}=fact_mat3;
        fN_all_comp{5}{4}=core_tens;
        fN_all_comp{5}{5}=size(fN_all{5});
        fN_all{5}=[];
        aifei=toc(fei);
        disp(['  Time for Tucker compression block 5::: ',num2str(aifei)])
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% block 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp=ten_mat_prod(pre_data_far_med{4}{6},{pre_data_far_med{1}{6},pre_data_far_med{2}{6},pre_data_far_med{3}{6}});
G_mn{6}(1:L,1:M,1:N)=(dx).*temp(1:L,1:M,1:N); G_mn{6}=G_mn{6}* ko^2;
temp_tens=zeros(2*LMN_new(1),2*LMN_new(2),LMN_new(3));
temp_tens(1:L,1:M,1:N) = G_mn{6};
% Cube 'L'
temp_tens(L+2+diffx:2*L+diffx,1:M,1:N)                                   = G_mn{6}(L:-1:2,1:M,1:N) * Gp_coeff_L(6,1);
% Cube 'M'
temp_tens(1:L,M+2+diffy:2*M+diffy,1:N)                                   = G_mn{6}(1:L,M:-1:2,1:N) * Gp_coeff_M(6,1);
% Cube 'N'
temp_tens(1:L,1:M,N+2+diffz:2*N+diffz)                                   = G_mn{6}(1:L,1:M,N:-1:2) * Gp_coeff_N(6,1);
% Cube 'LM'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,1:N)                   = G_mn{6}(L:-1:2,M:-1:2,1:N) * Gp_coeff_LM(6,1);
% Cube 'LN'
temp_tens(L+2+diffx:2*L+diffx,1:M,N+2+diffz:2*N+diffz)                   = G_mn{6}(L:-1:2,1:M,N:-1:2) * Gp_coeff_LN(6,1);
% Cube 'MN'
temp_tens(1:L,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz)                 = G_mn{6}(1:L,M:-1:2,N:-1:2) * Gp_coeff_MN(6,1);
% Cube 'LMN'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz) = G_mn{6}(L:-1:2,M:-1:2,N:-1:2) * Gp_coeff_LMN(6,1);
G_mn{6}=[];
% Obtain the FFTs of circulant
if (fl_no_fft == 1)
    fN_all{6} = temp_tens;
else
    tic
    fN_all{6} = fft_operator(temp_tens);
    disp(['  Time for FFT of block circulant tensor 6 ::: ',num2str(toc)])
    if fl_Tucker==1
        %%% Tucker compression and figuring out the computational overhead
        fei=tic;
        fN_all_comp{6}=cell(5,1);
        [fact_mat1,fact_mat2,fact_mat3,core_tens,mem_tens6,~] = Tucker(fN_all{6},tol_Tucker);
        fN_all_comp{6}{1}=fact_mat1;
        fN_all_comp{6}{2}=fact_mat2;
        fN_all_comp{6}{3}=fact_mat3;
        fN_all_comp{6}{4}=core_tens;
        fN_all_comp{6}{5}=size(fN_all{6});
        fN_all{6}=[];
        aifei=toc(fei);
        disp(['  Time for Tucker compression block 6::: ',num2str(aifei)])
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% block 7 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp=ten_mat_prod(pre_data_far_med{4}{7},{pre_data_far_med{1}{7},pre_data_far_med{2}{7},pre_data_far_med{3}{7}});
G_mn{7}(1:L,1:M,1:N)=(dx).*temp(1:L,1:M,1:N); G_mn{7}=G_mn{7}* ko^2;
temp_tens=zeros(2*LMN_new(1),2*LMN_new(2),LMN_new(3));
temp_tens(1:L,1:M,1:N) = G_mn{7};
% Cube 'L'
temp_tens(L+2+diffx:2*L+diffx,1:M,1:N)                                   = G_mn{7}(L:-1:2,1:M,1:N) * Gp_coeff_L(7,1);
% Cube 'M'
temp_tens(1:L,M+2+diffy:2*M+diffy,1:N)                                   = G_mn{7}(1:L,M:-1:2,1:N) * Gp_coeff_M(7,1);
% Cube 'N'
temp_tens(1:L,1:M,N+2+diffz:2*N+diffz)                                   = G_mn{7}(1:L,1:M,N:-1:2) * Gp_coeff_N(7,1);
% Cube 'LM'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,1:N)                   = G_mn{7}(L:-1:2,M:-1:2,1:N) * Gp_coeff_LM(7,1);
% Cube 'LN'
temp_tens(L+2+diffx:2*L+diffx,1:M,N+2+diffz:2*N+diffz)                   = G_mn{7}(L:-1:2,1:M,N:-1:2) * Gp_coeff_LN(7,1);
% Cube 'MN'
temp_tens(1:L,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz)                 = G_mn{7}(1:L,M:-1:2,N:-1:2) * Gp_coeff_MN(7,1);
% Cube 'LMN'
temp_tens(L+2+diffx:2*L+diffx,M+2+diffy:2*M+diffy,N+2+diffz:2*N+diffz) = G_mn{7}(L:-1:2,M:-1:2,N:-1:2) * Gp_coeff_LMN(7,1);

% Obtain the FFTs of circulant
if (fl_no_fft == 1)
    fN_all{7} = temp_tens;
else
    tic
    fN_all{7} = fft_operator(temp_tens);
    disp(['  Time for FFT of block circulant tensor 7 ::: ',num2str(toc)])
    if fl_Tucker==1
        %%% Tucker compression and figuring out the computational overhead
        fei=tic;
        fN_all_comp{7}=cell(5,1);
        [fact_mat1,fact_mat2,fact_mat3,core_tens,mem_tens7,~] = Tucker(fN_all{7},tol_Tucker);
        fN_all_comp{7}{1}=fact_mat1;
        fN_all_comp{7}{2}=fact_mat2;
        fN_all_comp{7}{3}=fact_mat3;
        fN_all_comp{7}{4}=core_tens;
        fN_all_comp{7}{5}=size(fN_all{7});
        fN_all{7}=[];
        aifei=toc(fei);
        disp(['  Time for Tucker compression block 7::: ',num2str(aifei)])
    end
end

disp(['  Time for computing circulant tensor ::: ',num2str(toc)])
% Remove unnecessary memory consuming elements
clear temp_tens;

memestimated = 7*16*8*LMN_new(1)*LMN_new(2)*LMN_new(3)/(1024*1024); % times 2 for real2cmplx
disp(['  Memory for storing original circulant tensor (MB) ::: ' , num2str(memestimated)]);

if fl_Tucker==1
    memory_compress = mem_tens1+mem_tens2+mem_tens3+mem_tens4+mem_tens5+mem_tens6+mem_tens7;
    memory_compress=memory_compress*16/(1024^2);
    disp(['Memory for storing compressed circulant tensors (MB) ::: ' , num2str(memory_compress)]);
end


% Normalizing linear basis functions
% % Attention::: Here has been added!!!
% G_mn(:,:,:,2:3) = G_mn(:,:,:,2:3) *(1/dx);
% G_mn(:,:,:,4) = G_mn(:,:,:,4) *(1/(dx^2));
% G_mn(:,:,:,5) = G_mn(:,:,:,5) *(1/(dx));
% G_mn(:,:,:,6:7) = G_mn(:,:,:,6:7) *(1/(dx^2));

% Self terms for sparse preconditioner

st_sparse_precon=[squeeze(G_mn{1}(1,1,1)) squeeze(G_mn{4}(1,1,1)) squeeze(G_mn{7}(1,1,1))];
G_mn{1}=[];  G_mn{4}=[]; G_mn{7}=[];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if fl_Tucker==0
    fN_all_comp=fN_all;
end
