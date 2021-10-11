clc;clear all;close all;
pre_define_the_path_for_folders

L=800;M=200;N=100;tol_Tucker=1e-8;
[fN_prestored_data] = store_Toep(1,L,M,N,tol_Tucker);
save('fN_prestored_data_800_200_100.mat', 'fN_prestored_data');

