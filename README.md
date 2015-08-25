This directory includes sources used in the following paper:

Ching-pei Lee and Dan Roth, Distributed Box-Constrained Quadratic Optimization for Dual Linear SVM, 2015.
You will be able to regenerate experiment results in the paper using this implementation.
However, results may be slightly different due to the randomness, the CPU speed,
and the load of your computer.

Please cite the above article if you find this tool useful. Please also read
the COPYRIGHT before using this tool.

The implementation is based on MPI LIBLINEAR.

If you have any questions, please contact:
Ching-pei Lee
leechingpei@gmail.com

Solvers
=======
The following solvers are supported.
-	0 -- quadratic box-constrained optimization with exact line search for L2-loss SVM
-	1 -- quadratic box-constrained optimization with exact line search for L1-loss SVM
-	2 -- quadratic box-constrained optimization with Armijo line search for L2-loss SVM
-	3 -- quadratic box-constrained optimization with Armijo line search for L1-loss SVM
-	4 -- DSVM-AVE/CoCoA for L2-loss SVM
-	5 -- DSVM-AVE/CoCoA for L1-loss SVM
-	6 -- Trust region Newton method for L2-loss SVM (primal)
-	7 -- DisDCA practical variant for L2-loss SVM
-	8 -- DisDCA practical variant for L1-loss SVM


Solver 6 is directly copied from MPI LIBLINEAR. Its documentation is available in
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/distributed-liblinear/
Zhuang, Yong, Chin, Wei-Sheng, Juan, Yu-Chin, and Lin, Chih-Jen. Distributed Newton method for regularized logistic regression. In Proceedings of The Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD), 2015.

Solvers 0-3 implement methods in
Lee, Ching-pei and Roth, Dan, Distributed Box-Constrained Quadratic Optimization for Dual Linear SVM, 2015.

Solvers 4-5 implement the DSVM-AVE method in
Pechyony, Dmitry, Shen, Libin, and Jones, Rosie. Solving large scale linear SVM with distributed block minimization. In Neural Information Processing Systems Workshop on Big Learning: Algorithms, Systems, and Tools for Learning at Scale, 2011.

Solvers 7-8 implement the DisDCA practical variant in
Yang, Tianbao. Trading computation for communication: Distributed stochastic dual coordinate ascent. In Advances in Neural Information Processing Systems 26, pp. 629–637, 2013.


Running Environment
===================
This code is supposed to be run on UNIX machines. The following
commands are required:

- g++
- make
- split

All methods require MPI libraries. Available implementations include

- OpenMPI
You can find the information about OpenMPI at

http://www.open-mpi.org/

- MPICH
You can find the information about MPICH at

http://www.mpich.org/


Usage
=====
To compile the code, type

	> make

To train a model distributedly, we provide a script "split.py" to split data into segments.
This scripts requires a file of the node list, and assume one I/O thread per node is used.
That is, we do not allow duplicated machines in the list file.
We provide an example of the node list in the file "machinelist"
Assume the original training file name is train_file.
If you are using an NFS system, then the segmented filenames will be train_file.00,
train_file.01,...
The enumeration starts from 0, and all files are of the same number of digits for enumeration.
If you are not using an NFS system, then specify this by the option -f 1.
To see the full usage of this script, type

	> python split.py

The segmented files will then be copied to the machines and all will have the same filename train_file.sub.

To run the package, type

	> mpirun -hostfile machinelist ./train

and see the usage.

