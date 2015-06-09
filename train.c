#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <mpi.h>
#include "linear.h"
#include <time.h>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static timespec start, end;
extern timespec io_time;

void print_null(const char *s) {}

void exit_with_help()
{
	if(mpi_get_rank() != 0)
		mpi_exit(1);
	printf(
	"Usage: train [options] training_set_file test_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 0)\n"
	"  for multi-class classification\n"
	"	 0 -- quadratic box-constrained optimization with exact line search for L2-loss SVM\n"
	"	 1 -- quadratic box-constrained optimization with exact line search for L1-loss SVM\n"
	"	 2 -- quadratic box-constrained optimization with Armijo line search for L2-loss SVM\n"
	"	 3 -- quadratic box-constrained optimization with Armijo line search for L1-loss SVM\n"
	"	 4 -- DSVM-AVE/CoCoA for L2-loss SVM\n"
	"	 5 -- DSVM-AVE/CoCoA for L1-loss SVM\n"
	"	 6 -- Trust region Newton method for L2-loss SVM (primal)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-f NFS_flag : if NFS_flag is true, then the data segments take the names in the form file_name.00, file_name.01,....(all with the same number of digits); if it is false, then each machine take sthe same file name (default 0)\n"
	"-q : quiet mode (no outputs)\n"
	"-o opt: optimal value. If specified, will terminate when primal/dual are both within epsilon of opt\n"
	"-e epsilon : set tolerance of termination criterion (default 0.01)\n"
	"  -s 0-5 :\n"
	"\tif opt is specified: primal and dual objective values are both within epsilon of opt\n"
	"\telse: primal - dual < epsilon\n"
	"  -s 6 :\n"
	"\tif opt is specified: primal objective values is within epsilon of opt\n"
	"\telse: default tron stopping criterion following liblinear\n"
	);
	mpi_exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"[rank %d] Wrong input format at line %d\n", mpi_get_rank(), line_num);
	mpi_exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, char *input_file_name_2);
void read_problem(const char *filename);
void read_problem_test(const char *filename);

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
double bias;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int global_l, global_n;

	char input_file_name[1024];
	char input_file_name_2[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name, input_file_name_2);
	
	clock_gettime(CLOCK_REALTIME, &start);
	read_problem(input_file_name);
	clock_gettime(CLOCK_REALTIME, &end);
	io_time = timediff(start,end);
	read_problem_test(input_file_name_2);
	error_msg = check_parameter(&prob,&param);

	global_l = prob.l;
	global_n = prob.n;
	mpi_allreduce_notimer(&global_l, 1, MPI_INT, MPI_SUM);
	mpi_allreduce_notimer(&global_n, 1, MPI_INT, MPI_MAX);
	prob.n = global_n;

	if(mpi_get_rank()==0)
		printf("#instance = %d, #feature = %d\n", global_l, global_n);
	if(error_msg)
	{
		if(mpi_get_rank()==0)
			fprintf(stderr,"ERROR: %s\n", error_msg);
		mpi_exit(1);
	}

	{
		model_=train(&prob, &param);
		if(save_model(model_file_name, model_))
		{
			fprintf(stderr,"[rank %d] can't save model to file %s\n", mpi_get_rank(), model_file_name);
			mpi_exit(1);
		}
		free_and_destroy_model(&model_);
	}
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);
	free(probtest.y);
	free(probtest.x);
	free(x_spacetest);

	MPI_Finalize();
	return 0;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, char *input_file_name_2)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout
	int nfs = 0;

	// default values
	param.solver_type = L2_QBOE;
	param.C = 1;
	param.eps = 1e-2;
	param.opt = -1;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'f':
				nfs = atoi(argv[i]);
				break;
			case 'o':
				param.opt = atof(argv[i]);
				break;
			case 's':
				param.solver_type = atoi(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'B':
				bias = atof(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;

			default:
				if(mpi_get_rank() == 0)
					fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc - 1)
		exit_with_help();
	if (nfs)
	{
		char tmp_fmt[8];
		sprintf(tmp_fmt,"%%s.%%0%dd", int(log10(mpi_get_size()))+1);
		sprintf(input_file_name,tmp_fmt, argv[i], mpi_get_rank());
		sprintf(input_file_name_2,tmp_fmt, argv[i+1], mpi_get_rank());
	}
	else
	{
		strcpy(input_file_name, argv[i]);
		strcpy(input_file_name_2, argv[i+1]);
	}

	if(i<argc-2)
		strcpy(model_file_name,argv[i+2]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"[rank %d] can't open input file %s\n",mpi_get_rank(),filename);
		mpi_exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	fclose(fp);
}

void read_problem_test(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"[rank %d] can't open input file %s\n",mpi_get_rank(),filename);
		mpi_exit(1);
	}

	probtest.l = 0;
	elements = 0;
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		probtest.l++;
	}
	rewind(fp);

	probtest.bias=bias;

	probtest.y = Malloc(double,probtest.l);
	probtest.x = Malloc(struct feature_node *,probtest.l);
	x_spacetest = Malloc(struct feature_node,elements+probtest.l);

	max_index = 0;
	j=0;
	for(i=0;i<probtest.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		probtest.x[i] = &x_spacetest[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		probtest.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_spacetest[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_spacetest[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_spacetest[j].index;

			errno = 0;
			x_spacetest[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(probtest.bias >= 0)
			x_spacetest[j++].value = probtest.bias;

		x_spacetest[j++].index = -1;
	}

	if(probtest.bias >= 0)
	{
		probtest.n=max_index+1;
		for(i=1;i<probtest.l;i++)
			(probtest.x[i]-2)->index = probtest.n;
		x_spacetest[j-2].index = probtest.n;
	}
	else
		probtest.n=max_index;

	fclose(fp);
}
