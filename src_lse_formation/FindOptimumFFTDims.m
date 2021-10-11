function nelem_xyz_new=FindOptimumFFTDims(nelem_xyz)
% This subroutine accepts the dimensions of 3D array that will be input to fftw.
% And it finds the number of elements that should be added to this array
% for optimum execution of fftw.
   
% global_com
% global fout

% tensor size: 2L, 2M, 2N
    
num_ints=zeros(1,6);num_reps=zeros(1,6);opt_pnt=zeros(3,6);
pnt_fin=zeros(1,3);tot_elems=zeros(1,3);diff_fin=zeros(1,3);
nelem_xyz_new=zeros(1,3);
tot_elems(1)=2*nelem_xyz(1); %%%need to change 
tot_elems(2)=2*nelem_xyz(2); %%%need to change 
tot_elems(3)=2*nelem_xyz(3); %%%need to change 
    
%allowed_int=40
allowed_int=100;
thres=30000;
    
all_ints=[2,3,5,7,11,13];
% for ii=1:3
%     if 


% Attention: This routine should be checked for further fft sizes
% for kk=1:3
%     if (tot_elems(kk) > thres)
%         disp(sprintf('Increase the threshold in optimum fft size module!!!'))
%         fprintf(fout,'%s\n','Increase the threshold in optimum fft size module!!!');
%         break
%     end
% end

% Finding the number of powers of integers smaller than threshold
num_maxpwr=15;

for kk=1:6 % loop over integers
    dum=0;
    for ll=0:num_maxpwr
        if (all_ints(kk)^ll < thres)
            %print*,all_ints(kk)**ll
            dum=dum+1;
        else
            break
        end
    end
    num_ints(kk)=dum; %31
end
    
%     !dum=product(num_ints,1)
%     !print*,'Total number of combinations that will be tested:::',dum
%     !dum=sum(num_ints)
%     !print*,'Total number of integers in list:::',dum
%     
%     ! Allocating pointer entries and filling
    
% for kk=1:6
%     all_combs_ents=zeros(num_ints(kk),kk);
% end
    
for kk=1:6 % loop over integers
    dum=0;
    for ll=0:num_maxpwr
        if (all_ints(kk)^ll < thres)
            dum=dum+1;
            all_combs_ents(dum,kk)=all_ints(kk)^ll;
        else
            break % goto 32
        end
    end
    dum=0; %32
end
    
%     ! This is due to the requirement 11^a*13^b (a+b=0 or 1)
%     ! Still not good (it could be 2) but sufficient for now
all_combs_ents(3:num_ints(5),5)=1;
all_combs_ents(3:num_ints(6),6)=1;
    

% Cartesian product part
    
for kk=1:5
    num_reps(kk)=prod(num_ints(kk+1:6));
end
num_reps(6)=1;
    
num_pnts=prod(num_ints);
pnts=zeros(num_pnts,6);
    
for ll=1:6
    for kk=1:num_pnts/num_reps(ll)
        % pp=nint(mod(real(kk),real(num_reps(ll))))
        pp=mod(kk,num_ints(ll));
        % print*,ll,kk,pp
        if (pp == 0)
            pp=num_ints(ll);
        end
        pnts((kk-1)*num_reps(ll)+1:kk*num_reps(ll),ll)=all_combs_ents(pp,ll);
    end
end
    
% Finding the optimum point near to the given dimension size
diff_fin(1:3)=0;
for ll=1:3
    diff=thres;
    for kk=1:num_pnts
        if (prod(pnts(kk,:)) < thres)
            if (prod(pnts(kk,:)) > tot_elems(ll) && prod(pnts(kk,:)) < tot_elems(ll)+ allowed_int && mod(prod(pnts(kk,:)),2) == 0 ) %%%need to change 
                dum=prod(pnts(kk,:));
                if (dum < diff)
                    % print*,ll,dum,pnts(kk,:)
                    % print*,dum
                    diff=dum;
                    diff_fin(ll)=dum-tot_elems(ll);
                    pnt_fin(ll)=dum;
                    opt_pnt(ll,1:6)=pnts(kk,1:6);
                end
                % goto 69
            end
        end
    end
    % 69     print*, 'Point was found!!!'
end
    
% Deallocations
clear pnts all_combs_ents
    
% for kk=1:3
%     if (diff_fin(kk) == 0)
%         disp(sprintf('Optimum combination could not be located for dimension:: %d',kk))
%         disp(sprintf('Either increase threshold or do not make any addition'))
%         fpritf(fout,'%s %d\n', 'Optimum combination could not be located for dimension::',kk);
%         fpritf(fout,'%s\n','Either increase threshold or do not make any addition');
%     end
% end
          
nelem_xyz_new(1)=(tot_elems(1)+diff_fin(1))/2; %%%need to change 
nelem_xyz_new(2)=(tot_elems(2)+diff_fin(2))/2; %%%need to change 
nelem_xyz_new(3)=(tot_elems(3)+diff_fin(3))/2; %%%need to change 
    
diff_fin=diff_fin/2;
% disp(sprintf('Number of extra elements for opt. FFT :: %d%d%d',diff_fin))
% disp(sprintf('# of boxes along x, y, and z directions (new) :: %f%f%f',  nelem_xyz_new(1), nelem_xyz_new(2), nelem_xyz_new(3)))
% fprintf(fout,'%s %d %d %d\n','Number of extra elements for opt. FFT ::',diff_fin);
% fprintf(fout,'%s %d %d %d\n','# of boxes along x, y, and z directions (new) ::',  nelem_xyz_new(1), nelem_xyz_new(2), nelem_xyz_new(3));

%     !print*,'opt_pnt1', opt_pnt(1,:)
%     !print*,'opt_pnt2', opt_pnt(2,:)
%     !print*,'opt_pnt3', opt_pnt(3,:)

end

