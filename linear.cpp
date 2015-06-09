#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"
#include <mpi.h>
#include <set>
#include <map>
#include <time.h>

double opt_val;
double eps;
double best_primal;
double* best_w;
enum{L2R_L2LOSS_SVC_DUAL, L2R_L1LOSS_SVC_DUAL};

static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}

static timespec comm_time, comp_time, idle_time, total_comp_time;
static timespec start, end, start1, end1;
static timespec all_start;
timespec io_time;


struct feature_node *x_spacetest;
struct problem probtest;
double global_pos_label;

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

template<typename T>
void mpi_allreduce(T *buf, const int count, MPI_Datatype type, MPI_Op op)
{
	std::vector<T> buf_reduced(count);

	clock_gettime(CLOCK_REALTIME, &end1);
	timeadd(&comp_time, timediff(start1,end1));
	clock_gettime(CLOCK_REALTIME, &start1);
	MPI_Barrier(MPI_COMM_WORLD);
	clock_gettime(CLOCK_REALTIME, &end1);
	timeadd(&idle_time, timediff(start1,end1));

	clock_gettime(CLOCK_REALTIME, &start1);
	MPI_Allreduce(buf, buf_reduced.data(), count, type, op, MPI_COMM_WORLD);
	for(int i=0;i<count;i++)
		buf[i] = buf_reduced[i];
	clock_gettime(CLOCK_REALTIME, &end1);
	timeadd(&comm_time, timediff(start1,end1));

	clock_gettime(CLOCK_REALTIME, &start1);
	MPI_Barrier(MPI_COMM_WORLD);
	clock_gettime(CLOCK_REALTIME, &end1);
	timeadd(&idle_time, timediff(start1,end1));

	clock_gettime(CLOCK_REALTIME, &start1);
}

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	if(mpi_get_rank()!=0)
		return;
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif
static double testpredict(double *w, int n)
{
	static long global_l = 0;
	int idx;
	if (global_l == 0)
	{
		global_l = probtest.l;
		mpi_allreduce_notimer(&global_l, 1, MPI_LONG, MPI_SUM);
	}
	double correct = 0;
	for (int i=0;i<probtest.l;i++)
	{
		const feature_node *lx=probtest.x[i];
		double dec = 0;
		for(; (idx=lx->index)!=-1; lx++)
			if(idx<=n)
				dec += w[idx-1]*lx->value;
		if (probtest.y[i] != global_pos_label)
			dec *= -1.0;
		if (dec > 0)
			correct += 1.0;
	}

	mpi_allreduce_notimer(&correct, 1, MPI_DOUBLE, MPI_SUM);
	return correct/(double)global_l;
}

static int compute_fun(double* w, double* alpha, double C, int L2, const problem *prob)
{
	static long iter = 0;
	iter++;
	clock_gettime(CLOCK_REALTIME, &end1);
	clock_gettime(CLOCK_REALTIME, &end);
	timeadd(&total_comp_time, timediff(all_start, end));
	timeadd(&comp_time, timediff(start1,end1));
	int i;
	int conv = 0;
	int l = prob->l;
	int w_size = prob->n;
	double reg = 0;
	double xi = 0;
	double dual = 0;
	for (i=0;i<w_size;i++)
		reg += w[i]*w[i];
	if (L2)
		for (i=0;i<l;i++)
			dual += alpha[i] * alpha[i] * 0.25 / C;

	for (i=0;i<l;i++)
	{
		double d = 0;
		feature_node *s=prob->x[i];
		while(s->index!=-1)
		{
			d += w[s->index-1] * s->value;
			s++;
		}
		d = 1 - d * prob->y[i];
		if (d > 0)
		{
			if (L2)
				xi += d * d;
			else
				xi += d;
		}
		dual -= alpha[i];
	}
	mpi_allreduce_notimer(&xi, 1, MPI_DOUBLE, MPI_SUM);
	mpi_allreduce_notimer(&dual, 1, MPI_DOUBLE, MPI_SUM);
	double f = reg / 2.0 + C * xi;
	if (f < best_primal)
	{
		best_primal = f;
		memcpy(best_w, w, sizeof(double) * w_size);
	}
	dual = dual + reg / 2.0;
	double acc = testpredict(best_w, w_size);
	if (opt_val >= 0)
	{
		if (fabs((fabs(dual) - fabs(opt_val))/opt_val) <= eps && fabs((fabs(f) - fabs(opt_val))/opt_val) <= eps)
			conv = 1;
	}
	else
	{
		if (fabs(dual + f) < eps)
			conv = 1;
	}
	info("Iter %d Dual %20.15e Primal %20.15e Time %g Accuracy %g\n", iter, dual, f, double(total_comp_time.tv_sec) + double(total_comp_time.tv_nsec)/double(1000000000), acc*100);
	mpi_allreduce_notimer(&conv, 1, MPI_INT, MPI_MAX);
	clock_gettime(CLOCK_REALTIME, &all_start);
	clock_gettime(CLOCK_REALTIME, &start1);
	return conv;
}

class l2r_l2_svc_fun: public function
{
public:
	l2r_l2_svc_fun(const problem *prob, double *C);
	~l2r_l2_svc_fun();

	double fun(double *w, int &conv);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

protected:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	I = new int[l];
	this->C = C;
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] D;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w, int &conv)
{
	int i;
	double f=0, reg=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		reg += w[i]*w[i];
	reg /= 2.0;
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}
	mpi_allreduce(&f, 1, MPI_DOUBLE, MPI_SUM);
	f += reg;

	clock_gettime(CLOCK_REALTIME, &end1);
	clock_gettime(CLOCK_REALTIME, &end);
	timeadd(&total_comp_time, timediff(all_start, end));
	timeadd(&comp_time, timediff(start1,end1));
	
	double acc = testpredict(w, w_size);
	info("FUN %15.20e acc %g time %g\n",f, acc* 100 ,double(total_comp_time.tv_sec) + double(total_comp_time.tv_nsec)/double(1000000000));
	clock_gettime(CLOCK_REALTIME, &all_start);
	clock_gettime(CLOCK_REALTIME, &start1);
	if (opt_val >= 0 && fabs(f - opt_val) / opt_val <= eps)
	{
		conv = 1;
	}
	else
		conv = 0;

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);
	mpi_allreduce(g, w_size, MPI_DOUBLE, MPI_SUM);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	mpi_allreduce(Hs, w_size, MPI_DOUBLE, MPI_SUM);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_svc_fun::subXv(double *v, double *Xv)
{
	int i;
	feature_node **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)
static void solve_l2r_l1l2_svc(
	const problem *prob, double *w,
	double Cp, double Cn, int solver_type, double beta)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 100000;
	int max_inner_iter;
	int *index = new int[l];
	double *alpha = new double[l];
	double *alpha_orig = new double[l];
	double *alpha_inc = new double[l];
	double *w_orig = new double[w_size];
	double *allreduce_buffer;
	double lambda = 0;
	schar *y = new schar[l];
	int line_search= 0;
	int converged = 0;
	int out_iter = 0;
	double eta = 1;
	int L2 = 1;
	double sigma = 0.1;
	double back_track = 0.5;
	double one_over_log_back = 1/log(0.5);
	double log_two = log(2.0);
	double max_step;
	int reduce_length = w_size;
	max_inner_iter = 1;
	if (beta > 0)
	{
		eta = beta / double(mpi_get_size());
		allreduce_buffer = new double[reduce_length];
	}
	else
	{
		reduce_length += 3;
		allreduce_buffer = new double[reduce_length];
		if(beta == 0)
			line_search = 1;
		else
			line_search = 2;
	}

	// PG: projected gradient, for shrinking and stopping
	double PG;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		L2 = 0;
		lambda = 1e-3;
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	if (beta > 0) //DSVM-AVE does not guarantee PD matrix
		lambda = 0;

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)] + lambda;

		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			w[xi->index-1] += y[i]*alpha[i]*val;
			xi++;
		}
		index[i] = i;
	}

	while (!converged && out_iter < max_iter)
	{
		out_iter++;
		iter = 0;
		for (i=0;i<l;i++)
			alpha_orig[i] = alpha[i];
		for (i=0;i<w_size;i++)
			w_orig[i] = w[i];
		for (int inner = 0;inner<max_inner_iter;inner++)
		{
			for (i=0; i<l; i++)
			{
				int j = i+rand()%(l - i);
				swap(index[i], index[j]);
			}


			for (s=0; s<l; s++)
			{
				i = index[s];
				G = 0;
				schar yi = y[i];

				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					G += w[xi->index-1]*(xi->value);
					xi++;
				}
				G = G*yi-1;

				C = upper_bound[GETI(i)];
				G += alpha[i]*diag[GETI(i)];

				PG = 0;
				if (alpha[i] == 0)
				{
					if (G < 0)
						PG = G;
				}
				else if (alpha[i] == C)
				{
					if (G > 0)
						PG = G;
				}
				else
					PG = G;

				if(fabs(PG) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
					d = (alpha[i] - alpha_old)*yi;
					xi = prob->x[i];
					while (xi->index != -1)
					{
						w[xi->index-1] += d*xi->value;
						xi++;
					}
				}
			}

			iter++;
		}

		for (i=0;i<l;i++)
			alpha_inc[i] = alpha[i] - alpha_orig[i];
		if (line_search)
			for (i=0;i<w_size;i++)
				allreduce_buffer[i] = w[i] - w_orig[i];
		if (line_search)
		{
			max_step = INF;
			double sum_alpha_inc = 0;
			double alpha_square = 0;
			double alpha_square_alpha = 0;
			for (i=0;i<l;i++)
			{
				sum_alpha_inc += alpha_inc[i];
				alpha_square += alpha_inc[i]*alpha_inc[i]*diag[GETI(i)];
				alpha_square_alpha += alpha_inc[i]*alpha_orig[i]*diag[GETI(i)];
				if (alpha_inc[i] > 0)
					max_step = min(max_step, (upper_bound[GETI(i)] - alpha_orig[i]) / alpha_inc[i]);
				else if (alpha_inc[i] < 0)
					max_step = min(max_step, alpha_orig[i] / (-alpha_inc[i]));
			}
			allreduce_buffer[w_size] = sum_alpha_inc;
			allreduce_buffer[w_size + 1] = alpha_square;
			allreduce_buffer[w_size + 2] = alpha_square_alpha;
			mpi_allreduce(&max_step, 1, MPI_DOUBLE, MPI_MIN);
		}

		if (line_search)
			mpi_allreduce(allreduce_buffer, reduce_length, MPI_DOUBLE, MPI_SUM);
		else
		{
			mpi_allreduce(w, reduce_length, MPI_DOUBLE, MPI_SUM);
			int size = mpi_get_size();
			for (i=0;i<w_size;i++)
				w[i] /= size;
		}

		if (line_search)
		{
			double w_inc_square;
			double w_w_inc;
			w_w_inc = 0;
			w_inc_square = 0;
			for (i=0;i<w_size;i++)
			{
				w_inc_square += allreduce_buffer[i]*allreduce_buffer[i];
				w_w_inc += allreduce_buffer[i] * w_orig[i];
			}

			if (line_search == 1)
			{
				double opt = (allreduce_buffer[w_size] - allreduce_buffer[w_size+2] - w_w_inc)/(allreduce_buffer[w_size+1] + w_inc_square);
				eta = min(opt, max_step);
			}
			else if (line_search == 2)
			{
			
				double sum_alpha_inc = allreduce_buffer[w_size];
				double alpha_square = allreduce_buffer[w_size+1];
				double alpha_square_alpha = allreduce_buffer[w_size+2];
				double new_k = ceil(one_over_log_back*(log_two - log(w_inc_square + alpha_square) +  log((sigma - 1.0) * (w_w_inc + alpha_square_alpha - sum_alpha_inc))));
				int int_new_k = int(new_k);
				eta = powi(back_track, max(0, int_new_k));
			}
			for (i=0;i<w_size;i++)
				w[i] = w_orig[i] + eta * allreduce_buffer[i];
		}

		for (i=0;i<l;i++)
			alpha[i] = alpha_orig[i] + eta * alpha_inc[i];

		converged += compute_fun(w, alpha, Cp, L2, prob);
		clock_gettime(CLOCK_REALTIME, &start1);
		clock_gettime(CLOCK_REALTIME, &all_start);
	}

	int nSV = 0;
	for(i=0; i<l; i++)
		if(alpha[i] > 0)
			++nSV;
	mpi_allreduce_notimer(&nSV, 1, MPI_INT, MPI_SUM);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
	delete [] alpha_inc;
	delete [] alpha_orig;
	delete [] w_orig;
}

#undef GETI

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
//
// In distributed environment, we need to make sure that the order of labels
// are consistent. It is achieved by three steps. Please see the comments in
// ``group_classes.''
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int i;


	// Step 1. Each node collects its own labels.
	// If you have enormous number of labels, you can use std::unordered_set
	// (whose complexity is O(1)) to replace std::set (whose complexity is
	// O(log(n))). Because std::unordered_set needs a compiler supporting C++11,
	// we use std::set for a better compatibility. Similarly, you may want to
	// replace std::map with std::unordered_map.
	std::set<int> label_set;
	for(i=0;i<prob->l;i++)
		label_set.insert((int)prob->y[i]);

	// Step 2. All labels are sent to the first machine.
	if(mpi_get_rank()==0)
	{
		for(i=1;i<mpi_get_size();i++)
		{
			MPI_Status status;
			int size;
			MPI_Recv(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			std::vector<int> label_buff(size);
			MPI_Recv(label_buff.data(), size, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			for(int j=0; j<size; j++)
				label_set.insert(label_buff[j]);
		}
	}
	else
	{
		int size = (int)label_set.size();
		std::vector<int> label_buff(size);
		i = 0;
		for(std::set<int>::iterator this_label=label_set.begin();
				this_label!=label_set.end(); this_label++)
		{
			label_buff[i] = (*this_label);
			i++;
		}
		MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(label_buff.data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	// Step 3. The first machine broadcasts the global labels to other nodes, so that
	// the order of labels in each machine is consistent.
	int nr_class = (int)label_set.size();
	MPI_Bcast(&nr_class, 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::map<int, int> label_map;
	int *label = Malloc(int, nr_class);
	{
		if(mpi_get_rank()==0)
		{
			i = 0;
			for(std::set<int>::iterator this_label=label_set.begin();
					this_label!=label_set.end(); this_label++)
			{
				label[i] = (*this_label);
				i++;
			}
		}
		MPI_Bcast(label, nr_class, MPI_INT, 0, MPI_COMM_WORLD);
		for(i=0;i<nr_class;i++)
			label_map[label[i]] = i;
	}

	// The following codes are similar to the original LIBLINEAR
	int *start = Malloc(int, nr_class);
	int *count = Malloc(int, nr_class);
	for(i=0;i<nr_class;i++)
		count[i] = 0;
	for(i=0;i<prob->l;i++)
		count[label_map[(int)prob->y[i]]]++;

	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[label_map[(int)prob->y[i]]]] = i;
		++start[label_map[(int)prob->y[i]]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	double runtime[4];
	clock_gettime(CLOCK_REALTIME, &all_start);
	clock_gettime(CLOCK_REALTIME, &start1);
	total_comp_time = timediff(start,start);
	comp_time = timediff(start1,start1);
	idle_time = timediff(start1,start1);
	comm_time = timediff(start1,start1);
	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		case L2_TRON:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			double local_eps;
			if (opt_val < 0)
				local_eps = eps;
			else
				local_eps = 1e-30;
			TRON tron_obj(fun_obj, local_eps);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			clock_gettime(CLOCK_REALTIME, &end1);
			timeadd(&comp_time, timediff(start1,end1));

			break;
		}
		case L2_QBOE:
			solve_l2r_l1l2_svc(prob, w, Cp, Cn, L2R_L2LOSS_SVC_DUAL, 0);
			break;
		case L2_QBOA:
			solve_l2r_l1l2_svc(prob, w, Cp, Cn, L2R_L2LOSS_SVC_DUAL, -1);
			break;
		case L2_DSVM:
			solve_l2r_l1l2_svc(prob, w, Cp, Cn, L2R_L2LOSS_SVC_DUAL, 1);
			break;
		case L1_QBOE:
			solve_l2r_l1l2_svc(prob, w, Cp, Cn, L2R_L1LOSS_SVC_DUAL, 0);
			break;
		case L1_QBOA:
			solve_l2r_l1l2_svc(prob, w, Cp, Cn, L2R_L1LOSS_SVC_DUAL, -1);
			break;
		case L1_DSVM:
			solve_l2r_l1l2_svc(prob, w, Cp, Cn, L2R_L1LOSS_SVC_DUAL, 1);
			break;
		default:
		{
			if(mpi_get_rank() == 0)
				fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
		}
	}
	clock_gettime(CLOCK_REALTIME, &end);
	timeadd(&total_comp_time, timediff(all_start, end));
	runtime[0] = double(comp_time.tv_sec) + double(comp_time.tv_nsec)/double(1000000000);
	runtime[1] = double(comm_time.tv_sec) + double(comm_time.tv_nsec)/double(1000000000);
	runtime[2] = double(idle_time.tv_sec) + double(idle_time.tv_nsec)/double(1000000000);
	runtime[3] = double(io_time.tv_sec) + double(io_time.tv_nsec)/double(1000000000);
	mpi_allreduce_notimer(runtime, 4, MPI_DOUBLE, MPI_SUM);
	for (int i=0;i<4;i++)
		runtime[i] /= mpi_get_size();
	info("Computation Time: %g s, Sync Time: %g s, Communication Time: %g s, IO Time: %g s\n", runtime[0], runtime[1], runtime[2], runtime[3]);
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);
	opt_val = param->opt;
	eps = param->eps;
	best_w = new double[n];
	best_primal = INF;

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		int k;
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];

		// multi-class svm by Crammer and Singer
		{
			if(nr_class == 2)
			{
				model_->w=Malloc(double, w_size);
				int e0 = start[0]+count[0];
				k=0;
				global_pos_label = label[0];
				for(; k<e0; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;

				train_one(&sub_prob, param, &model_->w[0], weighted_C[0], weighted_C[1]);
			}
			else
			{
				model_->w=Malloc(double, w_size*nr_class);
				double *w=Malloc(double, w_size);
				for(i=0;i<nr_class;i++)
				{
					int si = start[i];
					int ei = si+count[i];
					global_pos_label = label[i];

					k=0;
					for(; k<si; k++)
						sub_prob.y[k] = -1;
					for(; k<ei; k++)
						sub_prob.y[k] = +1;
					for(; k<sub_prob.l; k++)
						sub_prob.y[k] = -1;

					train_one(&sub_prob, param, w, weighted_C[i], param->C);

					for(int j=0;j<w_size;j++)
						model_->w[j*nr_class+i] = w[j];
				}
				free(w);
			}

		}

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		free(sub_prob.y);
		free(weighted_C);
	}
	free(best_w);
	return model_;
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
	{
		return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}


static const char *solver_type_table[]=
{
"L2_QBOE", "L1_QBOE", "L2_QBOA", "L1_QBOA", "L2_DSVM", "L1_DSVM", "L2_TRON"}; /* solver_type */

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"[rank %d] unknown solver type.\n", mpi_get_rank());

				setlocale(LC_ALL, old_locale);
				free(model_->label);
				free(model_);
				free(old_locale);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"[rank %d] unknown text in model file: [%s]\n",mpi_get_rank(),cmd);
			setlocale(LC_ALL, old_locale);
			free(model_->label);
			free(model_);
			free(old_locale);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != L2_QBOE && param->solver_type != L2_QBOA && param->solver_type != L2_DSVM && param->solver_type != L1_QBOE && param->solver_type != L1_QBOA && param->solver_type != L1_DSVM && param->solver_type != L2_TRON)
		return "unknown solver type";

	return NULL;
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}


int mpi_get_rank()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;	
}

int mpi_get_size()
{
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;	
}

void mpi_exit(const int status)
{
	MPI_Finalize();
	exit(status);
}
