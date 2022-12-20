# SADE-AMSS  
The matlab code for SADE-AMSS  
SADE-AMSS  
------------------------------- Reference --------------------------------  
H. Gu, H. Wang, and Y. Jin, Surrogate-Assisted Differential Evolution with Adaptive Multi-Subspace Search  
for Large-Scale Expensive Optimization in IEEE Transcations on Evolutionary Computation.  
------------------------------- Copyright --------------------------------  
Copyright (c) 2022 HandingWangXD Group. Permission is granted to copy and use this code for research, noncommercial purposes,  
provided this copyright notice is retained and the origin of the code is cited.  
The code is provided "as is" and without any warranties, express or implied.  
---------------------------- Parameter setting ---------------------------  
maxd ---  100  --- Maximum number of variables at each subspace  
Ns   ---  200  --- Initial size of Arc  
tsn  ---  2*d  --- The number of individuals in the training set  
Np   ---   10  --- The size of population  
K    ---   20  --- The maximum number of subspaces in a generation  
Gm   ---    5  --- The maximum iterations of subspace optimization  
tes  ---   50  --- A pre-set cutoff generation  
tr   ---  500  --- A pre-set cutoff generation  
beta ---    2  --- The threshold for switching strategy  
   
This code is written by Haoran Gu.  
Email: xdu_guhaoran@163.com  
