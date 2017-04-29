/*
 * File:   main.c
 * Author: Chris Dahnken (dahnken<at>gmx.net)
 *
 * Created on August 18, 2010, 8:36 AM
 */

/*
 * Cornelius
 * ---------
 * 2016/10/14
 * A thread-parallel versatile Lanczos diagonalization for the 
 * Hubbard model.
 * 
 * 
 * 
 * The application allows for each hopping, chemical potential and interaction 
 * for each spin to be individually configured. Hence open, periodic and
 * antiperiodic boundary conditions, 2-band, 3-band and periodic Anderson models
 * are easily calculated.  
 * 
 * Very little thought was spent to improve the performance, since only a 
 * smart way of verifying ground state energies for modes cluster sizes was
 * needed. 
 * 
 * The Hamiltonian is represented by to spin-separate matrices for the 
 * hopping and one matrix diagonal for the interaction and chemical potentials.
 * The memory consumption is, thus, pretty small compared to a full/dense 
 * representation of the Hamiltonian. A sparse implementation of the 
 * hopping matrices is considered as a development target.
 * 
 * Currently, only the ground state energy is computed. Calculation of the 
 * ground state vector is considered as a development target.
 * 
 * The diagonalization of tridiagonal Lanczos-matrix is comfortably done 
 * with GSL, which should be available on most Linux systems.
 * 
 * Input file configuration:
 * The input file in the following format is accepted:
 * 
 * <number of sites>
 * <number of up electrons> <number of down electrons>
 * <number of hoppings>
 * <from site> <to site> <hopping value> 
 * <from site> <to site> <hopping value> 
 * ...
 * <from site> <to site> <hopping value> 
 * <interaction site 1> <chemical potential site 1 up> <chemical potential site 1 up>
 * <interaction site 2> <chemical potential site 2 up> <chemical potential site 2 up>
 * ...
 * <interaction site nsites> <chemical potential site nsites up> <chemical potential site nsites up>
 * 
 * Comments can be placed after the last in number of a give line and at
 * the end of the file.
 * 
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <mm_malloc.h>
#include <iostream>
#include <fstream>
// number of sites
int nsites = 12;
// number of up electrons
int neup = 6;
// number of down electrons
int nedo = 6;
// array for the up-states
int* sup;
// array for the down states
int* sdo;
// interaction strength
double U = 0;
// chemical potential (both spins the same)
double mu = 0;
// number of states = nstatesup*nstatesdo
long nstates;
// number of states with spin up
long nstatesup;
// number of states with spin down
long nstatesdo;
// maximum number of iterations
int maxiter = 200;
// convergence criterion
double convCrit = 0.000000000001;
// hamilton matrix for spin up (hoppings)
// this matrix is of size nstatesup^2
double** mup;
// hamilton matrix for spin down (hoppings)
// this matrix is of size nstatesdo^2
double** mdo;
// hamilton matrix diagonal (chemical potential and interaction)
// this matrix is of size nstatesup*nstatesdo
double* mdia;
long blockup = 30;
// number of hoppings
int nhoppings;
// site-to-site hopping matrix
int** hopping;
// hopping values
double* hoppingvalue;
// spin-up chemical potential for each site
double* mu_up;
// spin-down chemical potential for each site
double* mu_do;
// interaction for each site
double* interaction;

#define LDEBUG

using namespace std;

// Read in the configuration file and set up 
// the arrays.
void readConfigFromFile(char* filename) {
    ifstream cfile(filename);
    // read line 1 - number of sites
    cfile >> nsites;
    // allocate the fields depending on the number of sites only:
    mu_up = new double[nsites];
    mu_do = new double[nsites];
    interaction = new double[nsites];
    // read line 2 - number of electrons (up down)
    cfile >> neup >> nedo;
    // read line 3 - number of hoppings 
    cfile >> nhoppings;
    // allocate fields depending on the number of hoppings:
    hopping = new int*[nhoppings];
    hoppingvalue = new double[nhoppings];
    // allocate the hoppings
    for (int i = 0; i < nhoppings; i++) {
        hopping[i] = new int[2];
    }
    // read in all the hoppings
    for (int i = 0; i < nhoppings; i++) {
        cfile >> hopping[i][0] >> hopping[i][1] >> hoppingvalue[i];
    }
    // read in interactions and chemical potential for up and down
    for (int i = 0; i < nsites; i++) {
        cfile >> interaction[i] >> mu_up[i] >> mu_do[i];
    }
    cfile.close();
}

// Faculty function - long type should suffice
long faculty(long i) {
    if (i == 1 || i == 0) {
        return 1;
    } else {
        return faculty(i - 1) * i;
    }
}

// Compute the number of states per spin
long nStatesPerSpin(int nsites, int nelec) {
    return faculty(nsites) / faculty(nelec) / faculty(nsites - nelec);
}

void printConfig() {
    cout << "--------------------------------------------------------" << endl; 
    cout << "                       CORNELIUS                        " << endl;
    cout << "--------------------------------------------------------" << endl; 
    cout << "Number of sites            : " << nsites << endl;
    cout << "Number of up electrons     : " << neup << endl;
    cout << "Number of down electrons   : " << nedo << endl;
    cout << "Number of hoppings         : " << nhoppings << endl;
    for (int i = 0; i < nhoppings; i++) {
        cout << "hopping " << i << "    : " 
                << hopping[i][0] << " " << hopping[i][1] 
                << " " << hoppingvalue[i] << endl;
    }
    for (int i = 0; i < nsites; i++) {
        cout << "interaction/mu_up/mu_down " 
                << i << " : " << interaction[i] << " " 
                << mu_up[i] << " " << mu_do[i] << endl;
    }
    cout << "Size of up hopping matrix  : " 
            << nStatesPerSpin(nsites, neup) << endl;
    cout << "Size of down hopping matrix: " << nStatesPerSpin(nsites, nedo) 
            << endl;
    cout << "Size of diagonal matrix    : " 
            << nStatesPerSpin(nsites, neup)*nStatesPerSpin(nsites, nedo) 
            << endl;
}

void printState(int s, int ns) {
    for (int i = 0; i < ns; i++) {
        int t = (s & (1 << i));
        if (t == 0) {
            printf("0");
        } else {
            printf("1");
        }
    }
    printf("\n");
}

long timeInSec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long) (tv.tv_sec * 1000 + tv.tv_usec / 1000.0);
}

// getBitAt
// Get a bit in an integer
int inline getBitAt(int st, int pos) {
    return ((st & (1 << pos)) >> pos);
}

// popcount 
// Sum all the bits in an integer.
// This is offloaded to an intrinsic function calling
// the respective assembly instruction.
int inline popcount(int i) {
    return __builtin_popcount(i);
}

// setupBasis
// Fills an arrays with numbers between 0 and 2^L-1
// having nelec number of bits (which correspond to 
// electrons).
void setupBasis(int*& basis, int nsites, int nelec) {
    int bmin = (1 << nelec) - 1;
    int bmax = bmin << (nsites - nelec);
    int n = 0;
    for (int i = bmin; i <= bmax; i++) {
        if (popcount(i) == nelec) {
            basis[n] = i;
            n++;
        }
    }
    printf("set up %d states\n", n);
}

// matrixelementU
// computes the number of double occupations between
// up and down states u and d. 
// This can be done a lot quicker, but has not been a 
// bottle neck at this time.
double matrixelementU(int u, int d) {
    double ret = 0;
    for (int i = 0; i < nsites; i++)
        ret += interaction[i] * getBitAt(u, i) * getBitAt(d, i);
    return ret;
}

double matrixelementMu(int u, int d) {
    double ret = 0;
    for (int i = 0; i < nsites; i++)
        ret += mu_up[i] * getBitAt(u, i) + mu_do[i] * getBitAt(d, i);
    return ret;
}

//inline double comsign(int r, int a, int b) {
//    int l = a < b ? a : b;
//    //    int h = a >= b ? a : b;
//    unsigned int mask = ((1 << nsitesx) - 2) << l;
//    return -(((popcount(mask & r) % 2) << 1) - 1);
//}

// TODO :  review this method

inline double comsign2(int r, int a, int b) {
    a = a % nsites;
    b = b % nsites;
    int l = a < b ? a : b;
    int h = a >= b ? a : b;

    unsigned int mask = ((1 << h) - 1)-((1 << (l + 1)) - 1);
    double ret = -(((popcount(mask & r) % 2) << 1) - 1);
    return ret;
}

double matrixelementT(int l, int r) {
    double tmp = 0;
    int n;
    int m;
    int ra;
    int re;
    int s1;
    int s2;

    for (int i = 0; i < nhoppings; i++) {
        int n = hopping[i][0];
        int m = hopping[i][1];
        double cs = comsign2(r, n, m);

        int ra = (r - (1 << n)+(1 << m));
        int re = (ra == l);
        int s1 = ((r & (1 << n)) >> n);
        int s2 = ((r & (1 << m)) >> m);
        s2 = (1 - s2);
        tmp += hoppingvalue[i] * cs * s1 * s2 * re;

        int p = n;
        n = m;
        m = p;
        ra = (r - (1 << n)+(1 << m));
        re = (ra == l);
        s1 = ((r & (1 << n)) >> n);
        s2 = ((r & (1 << m)) >> m);
        s2 = (1 - s2);
        tmp += hoppingvalue[i] * cs * s1 * s2 * re;
    }
    return tmp;
}


// 2D

//double XXmatrixelementT(int l, int r) {
//    double tmp = 0;
//    int n;
//    int m;
//    int ra;
//    int re;
//    int s1;
//    int s2;
//    // first all hoppings in x direction OBC
//    // no sign required here
//    for (int j = 0; j < nsitesy; j++) {
//        for (int i = j * nsitesx; i < (j + 1) * nsitesx - 1; i++) {
//            int n = i;
//            int m = i + 1;
//            int ra = (r - (1 << n)+(1 << m));
//            int re = (ra == l);
//            int s1 = ((r & (1 << n)) >> n);
//            int s2 = ((r & (1 << m)) >> m);
//            s2 = (1 - s2);
//            tmp += -s1 * s2 * re;
//
//            n = i + 1;
//            m = i;
//            ra = (r - (1 << n)+(1 << m));
//            re = (ra == l);
//            s1 = ((r & (1 << n)) >> n);
//            s2 = ((r & (1 << m)) >> m);
//            s2 = (1 - s2);
//            tmp += -s1 * s2 * re;
//        }
//    }
//    // ... then all hoppings in y direction
//    // ... sign required
//    // make a mask that is the nsitesx-1 values
//    // between i and i+nsitesx
//    //    unsigned int mask = (((1 << nsitesx) - 1) - 1);
//    for (int i = 0; i < nsitesx * (nsitesy - 1); i++) {
//        // compute the sign
//        //        int sgn = 1 - 2 * (popcount((r & (mask << i)))&1);
//        int n = i;
//        int m = i + nsitesx;
//        double sgn = comsign(r, n, m);
//        //        printf("sign =  %f\n", sgn);
//        int ra = (r - (1 << n)+(1 << m));
//        int re = (ra == l);
//        int s1 = ((r & (1 << n)) >> n);
//        int s2 = ((r & (1 << m)) >> m);
//        s2 = (1 - s2);
//        tmp += -s1 * s2 * re * sgn;
//
//        n = i + nsitesx;
//        m = i;
//        ra = (r - (1 << n)+(1 << m));
//        re = (ra == l);
//        s1 = ((r & (1 << n)) >> n);
//        s2 = ((r & (1 << m)) >> m);
//        s2 = (1 - s2);
//        tmp += -s1 * s2 * re * sgn;
//    }
//#ifdef _PBC_
//
//
//#endif
//    // ... then all PBC boundary hoppings
//
//    return tmp;
//}

double matrixelementT1D(int l, int r) {
    double tmp = 0;
    int n;
    int m;
    int ra;
    int re;
    int s1;
    int s2;

    for (int i = 0; i < nsites - 1; i++) {
        n = i;
        m = i + 1;
        ra = (r - (1 << n)+(1 << m));
        re = (ra == l);
        s1 = ((r & (1 << n)) >> n);
        s2 = ((r & (1 << m)) >> m);
        s2 = (1 - s2);
        tmp += -s1 * s2 * re;

        n = i + 1;
        m = i;
        ra = (r - (1 << n)+(1 << m));
        re = (ra == l);
        s1 = ((r & (1 << n)) >> n);
        s2 = ((r & (1 << m)) >> m);
        s2 = (1 - s2);
        tmp += -s1 * s2 * re;
    }
#ifdef _DIMENSION1_PBC_
    double sgn;
    n = nsites - 1;
    m = 0;
    ra = (r - (1 << n)+(1 << m));
    re = (ra == l);
    s1 = ((r & (1 << n)) >> n);
    s2 = ((r & (1 << m)) >> m);
    s2 = (1 - s2);
    sgn = -(((popcount(((1 << (nsites - 1)) - 2) & r) % 2) << 1) - 1); // - 1;
    tmp += -s1 * s2 * re * sgn;
    //    if (re) {
    //        printState(ra, nsites);
    //        printState(l, nsites);
    //        printf("%f \n", sgn);
    //    }

    n = 0;
    m = nsites - 1;
    ra = (r - (1 << n)+(1 << m));
    re = (ra == l);
    s1 = ((r & (1 << n)) >> n);
    s2 = ((r & (1 << m)) >> m);
    s2 = (1 - s2);
    sgn = -(((popcount(((1 << (nsites - 1)) - 2) & r) % 2) << 1) - 1); // - 1;
    tmp += -s1 * s2 * re * sgn;
    //    if (re) {
    //        printState(ra, nsites);
    //        printState(l, nsites);
    //        printf("%f \n", sgn);
    //    }
#endif
    return tmp;
}

double matrixelementT2(int l, int r) {
    double tmp = 0;
    for (int i = 0; i < nsites - 1; i++) {
        int n = i;
        int m = i + 1;

        int xn = (1 << n);
        int xm = (1 << m);

        int ra = (r - xn + xm);
        int re = (ra == l);
        int s1 = ((r & xn) >> n);
        int s2 = ((r & xm) >> m);
        s2 = (1 - s2);
        tmp += hoppingvalue[i] * s1 * s2 * re;

        ra = (r - xm + xn);
        re = (ra == l);
        s1 = ((r & xm) >> m);
        s2 = ((r & xn) >> n);
        s2 = (1 - s2);
        tmp += hoppingvalue[i] * s1 * s2 * re;
    }
    return tmp;
}

double matrixelement(int lu, int ld, int ru, int rd) {
    double r = 0;
    if (ld == rd && lu == ru) {
        r = matrixelementU(lu, ld);
        r += matrixelementMu(lu, ld);
    } else if (lu == ru) {
        r = matrixelementT(ld, rd);
    } else if (ld == rd) {
        r = matrixelementT(lu, ru);
    } else {
        r = 0;
    }
    return r;
}

void QPlusHTimesC1(double* &q, double* &c) {
#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long j = 0; j < nstatesdo; j++) {
            int p = i * nstatesup + j;
            double m = 0;
            for (long k = 0; k < nstatesup; k++) {
                for (long l = 0; l < nstatesdo; l++) {
                    int n = k * nstatesup + l;
                    m += matrixelement(sup[i], sdo[j], sup[k], sdo[l]) * c[n];
                }
            }
            q[p] += m;
        }
    }
}

// exploiting spin symmetry
void QPlusHTimesC2(double* &q, double* &c) {
#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long j = 0; j < nstatesdo; j++) {
            int p = i * nstatesup + j;
            double m = 0;
            for (long k = 0; k < nstatesup; k++) {
                int n = k * nstatesup + j;
                m += matrixelementT(sup[i], sup[k]) * c[n];
            }
            q[p] += m;
        }
    }
#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long j = 0; j < nstatesdo; j++) {
            int p = i * nstatesup + j;
            double m = 0;
            for (long l = 0; l < nstatesdo; l++) {
                int n = i * nstatesup + l;
                m += matrixelementT(sdo[j], sdo[l]) * c[n];
            }
            q[p] += m;
        }
    }
#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long j = 0; j < nstatesdo; j++) {
            int p = i * nstatesup + j;
            double m = 0;
            m += matrixelementU(sup[i], sdo[j]) * c[p];
            m += matrixelementMu(sup[i], sdo[j]) * c[p];
            q[p] += m;
        }
    }
}

void QPlusHTimesC3(double* &q, double* &c) {
    long time1, time2;

    printf("MatVec Hopping Up: ");
    time1 = timeInSec();
#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long j = 0; j < nstatesdo; j++) {
            int p = i * nstatesup + j;
            double m = 0;
            for (long k = 0; k < nstatesup; k++) {
                int n = k * nstatesup + j;
                m += mup[i][k] * c[n];
            }
            q[p] += m;
        }
    }
    time2 = timeInSec();
    printf("%ld\n", time2 - time1);

    printf("MatVec Hopping Down: ");
    time1 = timeInSec();
#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long j = 0; j < nstatesdo; j++) {
            int p = i * nstatesup + j;
            double m = 0;
            for (long l = 0; l < nstatesdo; l++) {
                int n = i * nstatesup + l;
                m += mdo[j][l] * c[n];
            }
            q[p] += m;
        }
    }
    time2 = timeInSec();
    printf("%ld\n", time2 - time1);

    printf("MatVec Diagonal: ");
    time1 = timeInSec();
#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long j = 0; j < nstatesdo; j++) {
            int p = i * nstatesup + j;
            double m = 0;
            m += mdia[p] * c[p];
            q[p] += m;
        }
    }
    time2 = timeInSec();
    printf("%ld\n", time2 - time1);
}

void QPlusHTimesC4(double* &q, double* &c) {
    long time1, time2;

    printf("MatVec Hopping Down: ");
    time1 = timeInSec();
#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long j = 0; j < nstatesdo; j++) {
            int p = i * nstatesup + j;
            double m = 0;
            for (long l = 0; l < nstatesdo; l++) {
                int n = i * nstatesup + l;
                m += mdo[j][l] * c[n];
            }
            q[p] += m;
        }
    }
    time2 = timeInSec();
    printf("%ld\n", time2 - time1);

    printf("MatVec Hopping Up: ");
    time1 = timeInSec();
    for (long kk = 0; kk < nstatesup; kk += blockup) {
#pragma omp parallel for
        for (long i = 0; i < nstatesup; i++) {
            for (int j = 0; j < nstatesdo; j++) {
                long p = i * nstatesup + j;
                double m = 0;
                int ub = fmin(kk + blockup, nstatesup);
                for (long k = kk; k < ub; k++) {
                    long n = k * nstatesup + j;
                    m += mup[i][k] * c[n];
                }
                q[p] += m;
            }
        }
    }

    time2 = timeInSec();
    printf("%ld\n", time2 - time1);

    printf("MatVec Diagonal: ");
    time1 = timeInSec();
#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long j = 0; j < nstatesdo; j++) {
            int p = i * nstatesup + j;
            double m = 0;
            m += mdia[p] * c[p];
            q[p] += m;
        }
    }
    time2 = timeInSec();
    printf("%ld\n", time2 - time1);
}

void solvetridiag(double &gse, double* &gsv, double* &alpha, double* &beta, int nIterations){
        gsl_matrix *mm = gsl_matrix_alloc(nIterations, nIterations);

        for (int i = 0; i < nIterations; i++) {
            for (int j = 0; j < nIterations; j++) {
                gsl_matrix_set(mm, i, j, 0);
            }
        }

        for (int i = 0; i < nIterations; i++) {
            gsl_matrix_set(mm, i, i, alpha[i]);
        }

        for (int i = 1; i < nIterations; i++) {
            gsl_matrix_set(mm, i - 1, i, beta[i - 1]);
            gsl_matrix_set(mm, i, i - 1, beta[i - 1]);
        }

        gsl_vector *eval = gsl_vector_alloc(nIterations);
        gsl_matrix *evec = gsl_matrix_alloc(nIterations, nIterations);

        gsl_eigen_symmv_workspace *w =
                gsl_eigen_symmv_alloc(nIterations);

        gsl_eigen_symmv(mm, eval, evec, w);
        gsl_eigen_symmv_free(w);
        gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);
        

        double* d = new double[nIterations];
//        for (int i = 0; i < nIterations; i++)
//            d[i] = gsl_vector_get(eval, i);
        gse=gsl_vector_get(eval,0);
//        gsl_vector_view v= gsl_matrix_column(evec,0);
        for(int i=0;i< nIterations;i++)
            gsv[i]=gsl_matrix_get(evec,0,i);
        gsl_matrix_free(mm);
        printf("Eigenvalue = %g\n", gse);
        // ------------ SolverTriDiagonal end
        delete(d);

}


void lanczos2() {
    int j;
    int nIterations = 0;
    double* c;
    double* q;
    double dnew;
    double* evec;
    double dold = 10000000;
    double* beta = new double[maxiter];
    double* alpha = new double[maxiter];
    c = new double[nstates];
    q = new double[nstates];
    long time1, time2;
    
#pragma omp parallel for
    for (long i = 0; i < nstates; i++) {
        c[i] = 1.0 / sqrt((double) nstates);
        q[i] = 0;
    }
    beta[0] = 0;
    j = 0;

    for (int n = 0; n < maxiter - 1; n++) {
        time1 = timeInSec();

        if (j != 0) {
#pragma omp parallel for
            for (long i = 0; i < nstates; i++) {
                double t = c[i];
                c[i] = q[i] / beta[j];
                q[i] = -beta[j] * t;
            }
        }
        QPlusHTimesC3(q, c);
        j++;

        double tmp = 0;
#pragma omp parallel for reduction(+:tmp)
        for (long i = 0; i < nstates; i++) {
            tmp += c[i] * q[i];
        }
        alpha[j] = tmp;
#pragma omp parallel for
        for (long i = 0; i < nstates; i++) {
            q[i] = q[i] - alpha[j] * c[i];
        }
        tmp = 0;
#pragma omp parallel for reduction(+:tmp)
        for (long i = 0; i < nstates; i++) {
            tmp += q[i] * q[i];
        }
        beta[j] = sqrt(tmp);
        nIterations = j;
        printf("beta[%d]=%e\n", j, beta[j]);

        // ------------ SolverTriDiagonal begin
        evec=new double[nIterations];
        solvetridiag(dnew,evec,alpha,beta,nIterations);
        // -------------- Tridiag solver end
        printf("Difference %e - %e = %e\n", dnew, dold, fabs(dnew - dold));
        if (j > 5) {
            if ((fabs(dnew - dold) < convCrit)) {
                break;
            }
            dold=dnew;
            free(evec);
        }
        
        time2 = timeInSec();
        printf("Iteration time in sec %ld\n", (time2 - time1));

    }
    delete(c);
    delete(q);
    delete(alpha);
    delete(beta);
    //    return 0;
}



void lanczos() {
    int j;
    int nIterations = 0;
    double* c;
    double* q;
    double* d;
    double dold = 10000000;
    double* beta = new double[maxiter];
    double* alpha = new double[maxiter];
    c = new double[nstates];
    q = new double[nstates];
    long time1, time2;
    
#pragma omp parallel for
    for (long i = 0; i < nstates; i++) {
        c[i] = 1.0 / sqrt((double) nstates);
        q[i] = 0;
    }
    beta[0] = 0;
    j = 0;

    for (int n = 0; n < maxiter - 1; n++) {
        time1 = timeInSec();

        if (j != 0) {
#pragma omp parallel for
            for (long i = 0; i < nstates; i++) {
                double t = c[i];
                c[i] = q[i] / beta[j];
                q[i] = -beta[j] * t;
            }
        }
        QPlusHTimesC4(q, c);
        j++;

        double tmp = 0;
#pragma omp parallel for reduction(+:tmp)
        for (long i = 0; i < nstates; i++) {
            tmp += c[i] * q[i];
        }
        alpha[j] = tmp;
#pragma omp parallel for
        for (long i = 0; i < nstates; i++) {
            q[i] = q[i] - alpha[j] * c[i];
        }
        tmp = 0;
#pragma omp parallel for reduction(+:tmp)
        for (long i = 0; i < nstates; i++) {
            tmp += q[i] * q[i];
        }
        beta[j] = sqrt(tmp);
        nIterations = j;
        printf("beta[%d]=%e\n", j, beta[j]);

        // ------------ SolverTriDiagonal begin
        gsl_matrix *mm = gsl_matrix_alloc(nIterations, nIterations);

        for (int i = 0; i < nIterations; i++) {
            for (int j = 0; j < nIterations; j++) {
                gsl_matrix_set(mm, i, j, 0);
            }
        }

        for (int i = 0; i < nIterations; i++) {
            gsl_matrix_set(mm, i, i, alpha[i]);
        }

        for (int i = 1; i < nIterations; i++) {
            gsl_matrix_set(mm, i - 1, i, beta[i - 1]);
            gsl_matrix_set(mm, i, i - 1, beta[i - 1]);
        }

        gsl_vector *eval = gsl_vector_alloc(nIterations);
        gsl_matrix *evec = gsl_matrix_alloc(nIterations, nIterations);

        gsl_eigen_symmv_workspace *w =
                gsl_eigen_symmv_alloc(nIterations);

        gsl_eigen_symmv(mm, eval, evec, w);
        gsl_eigen_symmv_free(w);
        gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);
        

        double* d = new double[nIterations];
        for (int i = 0; i < nIterations; i++)
            d[i] = gsl_vector_get(eval, i);
        gsl_matrix_free(mm);

        printf("Eigenvalue = %g\n", d[0]);
        // ------------ SolverTriDiagonal end

        printf("Difference %e - %e = %e\n", d[0], dold, fabs(d[0] - dold));
        if (j > 5) {
            if ((fabs(d[0] - dold) < convCrit)) {
                break;
            }
        }
        dold = d[0];
        delete(d);
        time2 = timeInSec();
        printf("Iteration time in sec %ld\n", (time2 - time1));

    }
    delete(c);
    delete(q);
    delete(alpha);
    delete(beta);
    //    return 0;
}

void setupMatrix() {
    long time1, time2;
    time1 = timeInSec();

#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long k = 0; k < nstatesup; k++) {
            mup[i][k] = matrixelementT(sup[i], sup[k]);
        }
    }
#pragma omp parallel for
    for (long j = 0; j < nstatesdo; j++) {
        for (long l = 0; l < nstatesdo; l++) {
            mdo[j][l] = matrixelementT(sdo[j], sdo[l]);
        }
    }
#pragma omp parallel for
    for (long i = 0; i < nstatesup; i++) {
        for (long j = 0; j < nstatesdo; j++) {
            int p = i * nstatesup + j;
            mdia[p] = matrixelementU(sup[i], sdo[j]);
            mdia[p] += matrixelementMu(sup[i], sdo[j]);
        }
    }
    //    for (int i = 0; i < nstatesup; i++) {
    //        for (int j = 0; j < nstatesup; j++) {
    //            printf("%f ", mup[i][j]);
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    for (int i = 0; i < nstatesdo; i++) {
    //        for (int j = 0; j < nstatesdo; j++) {
    //            printf("%f ", mdo[i][j]);
    //        }
    //        printf("\n");
    //    }
    time2 = timeInSec();
    printf("Setup: %ld\n", time2 - time1);

}

void init() {
    nstatesup = nStatesPerSpin(nsites, neup);
    nstatesdo = nStatesPerSpin(nsites, nedo);
    nstates = nstatesup*nstatesdo;
    sup = new int[nstatesup];
    sdo = new int[nstatesdo];
    mup = new double*[nstatesup];
    mdo = new double*[nstatesdo];
    mdia = new double[nstates];

    setupBasis(sup, nsites, neup);
    setupBasis(sdo, nsites, nedo);

    for (int i = 0; i < nstatesup; i++) {
        mup[i] = new double[nstatesup];
    }
    for (int i = 0; i < nstatesdo; i++) {
        mdo[i] = new double[nstatesdo];
    }
    setupMatrix();
}


/*
 *
 */
int main(int argc, char** argv) {

    readConfigFromFile(argv[1]);
    printConfig();
    printf("blockup = %ld\n", blockup);
    init();
    lanczos();
    return (EXIT_SUCCESS);
}


