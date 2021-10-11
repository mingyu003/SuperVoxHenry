clc; close all; clear all; format long e;
global fl_precon_type

% for debug only
global A_inv LL UU PP QQ RR Sch_sparse slct_decomp_sch fl_cholmod A_noninv W

disp('------------------------------------------------------------------')
disp('VoxHenry: Inductance Extraction Simulator for Voxelized Geometries')
disp('                                                                  ')
disp('          by Abdulkadir C. Yucel and Jacob K. White (MIT)         ')
disp('                                                                  ')
disp('                 Empowered by modules/ideas from                  ')
disp('   Athanasios G. Polimeridis and Ioannis P. Georgakis (Skoltech)  ')
disp('  Hakan Bagci (KAUST), Enrico Di Lorenzo (FastFieldSolvers S.R.L.)')
disp('------------------------------------------------------------------')

% -------------------------------------------------------------------------
%                  Add the Current Path to Workspace
% -------------------------------------------------------------------------

pre_define_the_path_for_folders

% -------------------------------------------------------------------------
%                  Simulation Parameters
% -------------------------------------------------------------------------

% input file
disp('Select input file:::');
inputfilelist = dir(['Input_files', filesep, '*.vhr']);
for filenum = 1:size(inputfilelist,1)
    disp([num2str(filenum), ' : ', inputfilelist(filenum).name]);
end
filenum = input('Open file number: ');
fileinname = inputfilelist(filenum).name;

er = 0;  % epsilon_r of conductors
inner_it = 100; outer_it = 10; tol=1e-12; % iterative solver inputs
prectol = 1e-1; % tolerance used in Schur GMRES inversion by 'schur_gmres' preconditioner
% use or do not use preconditioner
% valid values are: 'no_precond', 'schur_invert', 'schur_approx'
%fl_precon_type = 'no_precond';
%fl_precon_type = 'schur_approx';
fl_precon_type = 'schur_invert';
%fl_precon_type = 'schur_gmres';
%fl_precon_type = 'schur_invert_original';

% plotting
plot_currents_post_proc = 1; % if 1, plot the current densities in 3D
plot_option=1; % see the options of plotting in Visualization part
freq_curr_plot=2.5e9; % frequency for plotting currents


% -------------------------------------------------------------------------
%                  Inputs for Simulation
% -------------------------------------------------------------------------

disp(['Reading input file: ', fileinname]);
% read data from input file
% 'sigma_e' is a LxMxN array of conductivities. Zero means empty voxel, if not superconductor.
%           If superconductor, also 'lambdaL' must be zero to indicate an empty voxel.
% 'lambdaL' is a LxMxN array of London penetration depths, in case of superconductors.
%           If zero, and sigma_e is zero, means empty voxel. If there isn't any superconductor at all,
%           'lambdaL' is empty ( lambdaL = [] ) 
% 'freq' is the array of required simulation frequencies
% 'dx' is the voxel side dimension, in meters
% 'pnt_lft' is the cell array of port node relative positions, positive side 
% 'pnt_rght' is the cell array of port node relative positions, negative side 
% 'pnt_well_cond' is the cell array of grounded nodes relative positions 
[sigma_e, lambdaL, freq, dx, num_ports, pnt_lft, pnt_rght, pnt_well_cond] = pre_input_file(['Input_files', filesep, fileinname]);

% -------------------------------------------------------------------------
%                         Initialize stuff
% -------------------------------------------------------------------------

% sort and arrange frequency array
num_freq = length(freq);
if (issorted(freq) == 0) % not sorted
    freq=sort(freq)
end
freq_all = freq;
freq = freq(1); % currently do everything for the lowest freq

% generate EM constants (also frequency 'freq' dependent', e.g. 'omega')
EMconstants

% -------------------------------------------------------------------------
%                 Define EM Vars/Constants and Domain Parameters
% -------------------------------------------------------------------------

pre_define_structure_params

% output to screen a summary of the parameters
pre_print_out_inputs_generate_consts
    
% -------------------------------------------------------------------------
%                  Obtain Nodal Incidence Matrix
% -------------------------------------------------------------------------

disp('-----------------------------------------------------')
disp(['Generating panel IDs and Ae matrix...']);
tini = tic;

% Get the voxel2nodes matrix (each row is a non-empty panel, each of the six columns
% is a node ID in the order -x/+x/-y/+y/-z/+z)
%
% 'all_panels_ids' is the array of panel IDs, indexed per non-empty voxel. Voxel order
%                  is the MatLab/Octave native one for non-empty voxels of 'sigma_e'.
%                  Panel order is -x/+x/-y/+y/-z/+z. E.g. all_panels_ids(3,2) stores 
%                  the ID of the panel on the positive y face of the non-empty voxel number 3
[all_panels_ids, num_nodes] = lse_generate_panelIDs(idxS, L, M, N); 

[Ae, Ae_only_leaving, Ae_only_entering_bndry] = lse_compute_Ae_matrix(idxS, all_panels_ids, num_nodes);

tend = toc(tini);
disp(['Time for generating Ae mat & finding IDs of port nodes::: ' ,num2str(tend)]);
disp('-----------------------------------------------------')

sim_CPU_pre(1)=toc(tini); % CPU time for Ae

% ------------------------------------------------------------------------
%              Precomputation of LSE data
% ------------------------------------------------------------------------

tinisim = tic;
disp('-----------------------------------------------------')
disp(['Precomputing LSE data structures...'])

%
% circulant tensor
%

disp([' Generating circulant tensor...']);
tini = tic;

% no need to delay computation of FFT. FFT is a linear operator, and as the frequency-dependent
% part is not in the circulant tensor, we can just multiply later on by '1i*omega*mu'
fl_no_fft=0;
% note: must still multiply 'fN_all2' and 'st_sparse_precon2' by '1i*omega*mu'
% ( here we set ko = 1 in the the second parameter when calling 'lse_generate_circulant_tensor',
% so 'lse_generate_circulant_tensor' will not multiply by the actual 'ko'; 'ko' is frequency-dependent)
[fN_all2,st_sparse_precon2] = lse_generate_circulant_tensor(dx,1,L,M,N,fl_no_fft);

