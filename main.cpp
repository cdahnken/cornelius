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
 * smart way of verifying ground state energies for modest cluster sizes was
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

#define _SPARSE_

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
#include <inttypes.h>

// number of sites
uint64_t nsites = 12;
// number of up electrons
uint64_t neup = 6;
// number of down electrons
uint64_t nedo = 6;
// array for the up-states
uint64_t* sup;
// array for the down states
uint64_t* sdo;
// interaction strength
double U = 0;
// chemical potential (both spins the same)
double mu = 0;
// number of states = nstatesup*nstatesdo
uint64_t nstates;
// number of states with spin up
uint64_t nstatesup;
// number of states with spin down
uint64_t nstatesdo;
// maximum number of iterations
uint64_t maxiter = 200;
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
uint64_t blockup = 30;
// number of hoppings
uint64_t nhoppings;
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

#ifdef _SPARSE_
// how many values are in a given row?
uint32_t* sparse_nup;
uint32_t* sparse_ndo;

// pointers to the values
double** sparse_pvup;
double** sparse_pvdo;

// pointers to the columns
uint64_t** sparse_pcup;
uint64_t** sparse_pcdo;

// buffers
//double* sparse_vbuffer;
//uint64_t* sparse_cbuffer;

uint64_t sparse_buffersize=500;

#endif

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
    for (uint64_t i = 0; i < nhoppings; i++) {
        hopping[i] = new int[2];
    }
    // read in all the hoppings
    for (uint64_t i = 0; i < nhoppings; i++) {
        cfile >> hopping[i][0] >> hopping[i][1] >> hoppingvalue[i];
    }
    // read in interactions and chemical potential for up and down
    for (uint64_t i = 0; i < nsites; i++) {
        cfile >> interaction[i] >> mu_up[i] >> mu_do[i];
    }
    cfile.close();
}

// Faculty function - uint64_t type should suffice
uint64_t faculty(uint64_t i) {
    if (i == 1 || i == 0) {
        return 1;
    } else {
        return faculty(i - 1) * i;
    }
}

// Compute the number of states per spin
uint64_t nStatesPerSpin(uint64_t nsites, uint64_t nelec) {
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
    for (uint64_t i = 0; i < nhoppings; i++) {
        cout << "hopping " << i << " \t: " 
                << hopping[i][0] << " " << hopping[i][1] 
                << " " << hoppingvalue[i] << endl;
    }
    for (uint64_t i = 0; i < nsites; i++) {
        cout << "interaction/mu_up/mu_down " 
                << i << "\t: " << interaction[i] << " " 
                << mu_up[i] << " " << mu_do[i] << endl;
    }
    cout << "Size of up hopping matrix  : " 
            << nStatesPerSpin(nsites, neup) << endl;
    cout << "Size of down hopping matrix: " << nStatesPerSpin(nsites, nedo) 
            << endl;
    cout << "Size of diagonal matrix    : " 
            << nStatesPerSpin(nsites, neup)*nStatesPerSpin(nsites, nedo) 
            << endl;
    uint64_t tmp_nstates=nStatesPerSpin(nsites, neup)*nStatesPerSpin(nsites, nedo);
    uint64_t tmp_nstatesup=nStatesPerSpin(nsites, neup);
    uint64_t tmp_nstatesdo=nStatesPerSpin(nsites, nedo);

    cout << "Estimated memory footprint: "<< 
            (double(3)*double(tmp_nstates)
            + double(tmp_nstatesup)*double(tmp_nstatesup)
            + double(tmp_nstatesup)*double(tmp_nstatesup))*double(4)
            /double(1024)/double(1024.0)/double(1024.0)<<" GB"<<endl;
    cout << "--------------------------------------------------------" << endl; 
    // /double(1000000000.0)
}

void printState(uint64_t s, uint64_t ns) {
    for (uint64_t i = 0; i < ns; i++) {
        uint64_t t = (s & (1 << i));
        if (t == 0) {
            printf("0");
        } else {
            printf("1");
        }
    }
//    printf("\n");
}

double timeInSec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) ((tv.tv_sec * 1000 + tv.tv_usec / 1000.0))/1000.0;
}

// getBitAt
// Get a bit in an integer
uint64_t inline getBitAt(uint64_t st, uint64_t pos) {
    return ((st & (1 << pos)) >> pos);
}

// popcount 
// Sum all the bits in an integer.
// This is offloaded to an intrinsic function calling
// the respective assembly instruction.
uint64_t inline popcount(uint64_t i) {
    return __builtin_popcount(i);
}

// setupBasis
// Fills an arrays with numbers between 0 and 2^L-1
// having nelec number of bits (which correspond to 
// electrons).
void setupBasis(uint64_t*& basis, uint64_t nsites, uint64_t nelec) {
    uint64_t bmin = (1 << nelec) - 1;
    uint64_t bmax = bmin << (nsites - nelec);
    uint64_t n = 0;
    for (uint64_t i = bmin; i <= bmax; i++) {
        if (popcount(i) == nelec) {
            basis[n] = i;
            n++;
        }
    }
//    printf("set up %ld states\n", n);
    cout << "set up " <<n<< " states"<< endl;
}

// matrixelementU
// computes the number of double occupations between
// up and down states u and d. 
// This can be done a lot quicker, but has not been a 
// bottle neck at this time.
double matrixelementU(uint64_t u, uint64_t d) {
    double ret = 0;
    for (uint64_t i = 0; i < nsites; i++)
        ret += interaction[i] * getBitAt(u, i) * getBitAt(d, i);
    return ret;
}

double matrixelementMu(uint64_t u, uint64_t d) {
    double ret = 0;
    for (uint64_t i = 0; i < nsites; i++)
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

inline double comsign2(uint64_t r, uint64_t a, uint64_t b) {
    a = a % nsites;
    b = b % nsites;
    uint64_t l = a < b ? a : b;
    uint64_t h = a >= b ? a : b;

    uint64_t mask = ((1 << h) - 1)-((1 << (l + 1)) - 1);
    double ret = -(((popcount(mask & r) % 2) << 1) - 1);
    return ret;
}

double matrixelementT(uint64_t l, uint64_t r) {
    double tmp = 0;
    uint64_t n;
    uint64_t m;
    uint64_t ra;
    uint64_t re;
    uint64_t s1;
    uint64_t s2;

    for (uint64_t i = 0; i < nhoppings; i++) {
        uint64_t n = hopping[i][0];
        uint64_t m = hopping[i][1];
        double cs = comsign2(r, n, m);

        uint64_t ra = (r - (1 << n)+(1 << m));
        uint64_t re = (ra == l);
        uint64_t s1 = ((r & (1 << n)) >> n);
        uint64_t s2 = ((r & (1 << m)) >> m);
        s2 = (1 - s2);
        tmp += hoppingvalue[i] * cs * s1 * s2 * re;

        uint64_t p = n;
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

double matrixelementT1D(uint64_t l, uint64_t r) {
    double tmp = 0;
    uint64_t n;
    uint64_t m;
    uint64_t ra;
    uint64_t re;
    uint64_t s1;
    uint64_t s2;

    for (uint64_t i = 0; i < nsites - 1; i++) {
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

double matrixelementT2(uint64_t l, uint64_t r) {
    double tmp = 0;
    for (uint64_t i = 0; i < nsites - 1; i++) {
        uint64_t n = i;
        uint64_t m = i + 1;

        uint64_t xn = (1 << n);
        uint64_t xm = (1 << m);

        uint64_t ra = (r - xn + xm);
        uint64_t re = (ra == l);
        uint64_t s1 = ((r & xn) >> n);
        uint64_t s2 = ((r & xm) >> m);
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

double matrixelement(uint64_t lu, uint64_t ld, uint64_t ru, uint64_t rd) {
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
    for (uint64_t i = 0; i < nstatesup; i++) {
        for (uint64_t j = 0; j < nstatesdo; j++) {
            uint64_t p = i * nstatesup + j;
            double m = 0;
            for (uint64_t k = 0; k < nstatesup; k++) {
                for (uint64_t l = 0; l < nstatesdo; l++) {
                    uint64_t n = k * nstatesup + l;
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
    for (uint64_t i = 0; i < nstatesup; i++) {
        for (uint64_t j = 0; j < nstatesdo; j++) {
            uint64_t p = i * nstatesup + j;
            double m = 0;
            for (uint64_t k = 0; k < nstatesup; k++) {
                uint64_t n = k * nstatesup + j;
                m += matrixelementT(sup[i], sup[k]) * c[n];
            }
            q[p] += m;
        }
    }
#pragma omp parallel for
    for (uint64_t i = 0; i < nstatesup; i++) {
        for (uint64_t j = 0; j < nstatesdo; j++) {
            uint64_t p = i * nstatesup + j;
            double m = 0;
            for (uint64_t l = 0; l < nstatesdo; l++) {
                uint64_t n = i * nstatesup + l;
                m += matrixelementT(sdo[j], sdo[l]) * c[n];
            }
            q[p] += m;
        }
    }
#pragma omp parallel for
    for (uint64_t i = 0; i < nstatesup; i++) {
        for (uint64_t j = 0; j < nstatesdo; j++) {
            uint64_t p = i * nstatesup + j;
            double m = 0;
            m += matrixelementU(sup[i], sdo[j]) * c[p];
            m += matrixelementMu(sup[i], sdo[j]) * c[p];
            q[p] += m;
        }
    }
}

void QPlusHTimesC3(double* &q, double* &c) {
    double time1, time2;
#ifdef _SPARSE_
    printf("MatVec Hopping Up: ");
    time1 = timeInSec();
#pragma omp parallel for
    for (uint64_t i = 0; i < nstatesup; i++) {
        for (uint64_t j = 0; j < nstatesdo; j++) {
            uint64_t p = i * nstatesup + j;
            double m = 0;
//            for (uint64_t k = 0; k < nstatesup; k++) {
//                uint64_t n = k * nstatesup + j;
//                m += mup[i][k] * c[n];
//            }
            for(uint64_t n=0;n<sparse_nup[i];n++){
                m+=sparse_pvup[i][n]*c[sparse_pcup[i][n]*nstatesup+j];
//                printf("%d %d %d %f\n",i,j,sparse_pvup[i][n],sparse_pcup[i][n]);
            }    
            q[p] += m;
        }
    }
    time2 = timeInSec();
    printf("%f\n", time2 - time1);

    printf("MatVec Hopping Down: ");
    time1 = timeInSec();
#pragma omp parallel for
    for (uint64_t j = 0; j < nstatesdo; j++) {
        for (uint64_t i = 0; i < nstatesup; i++) {
            uint64_t p = i * nstatesup + j;
            double m = 0;
//            for (uint64_t l = 0; l < nstatesdo; l++) {
//                int n = i * nstatesup + l;
//                m += mdo[j][l] * c[n];
//            }
            for(uint64_t n=0;n<sparse_nup[j];n++){
                m+=sparse_pvdo[j][n]*c[i*nstatesup+sparse_pcdo[j][n]];
            }                
            q[p] += m;
        }
    }
    time2 = timeInSec();
    printf("%f\n", time2 - time1);

//    printf("MatVec Diagonal: ");
//    time1 = timeInSec();
//#pragma omp parallel for
//    for (uint64_t i = 0; i < nstatesup; i++) {
//        for (uint64_t j = 0; j < nstatesdo; j++) {
//            uint64_t p = i * nstatesup + j;
//            double m = 0;
//            m += mdia[p] * c[p];
//            q[p] += m;
//        }
//    }
//    time2 = timeInSec();
//    printf("%ld\n", time2 - time1);
#else
    printf("MatVec Hopping Up: ");
    time1 = timeInSec();
#pragma omp parallel for
    for (uint64_t i = 0; i < nstatesup; i++) {
        for (uint64_t j = 0; j < nstatesdo; j++) {
            uint64_t p = i * nstatesup + j;
            double m = 0;
            for (uint64_t k = 0; k < nstatesup; k++) {
                uint64_t n = k * nstatesup + j;
                m += mup[i][k] * c[n];
            }
            q[p] += m;
        }
    }
    time2 = timeInSec();
    printf("%f\n", time2 - time1);

    printf("MatVec Hopping Down: ");
    time1 = timeInSec();
#pragma omp parallel for
    for (uint64_t i = 0; i < nstatesup; i++) {
        for (uint64_t j = 0; j < nstatesdo; j++) {
            uint64_t p = i * nstatesup + j;
            double m = 0;
            for (uint64_t l = 0; l < nstatesdo; l++) {
                uint64_t n = i * nstatesup + l;
                m += mdo[j][l] * c[n];
            }
            q[p] += m;
        }
    }
    time2 = timeInSec();
    printf("%f\n", time2 - time1);

//    printf("MatVec Diagonal: ");
//    time1 = timeInSec();
//#pragma omp parallel for
//    for (uint64_t i = 0; i < nstatesup; i++) {
//        for (uint64_t j = 0; j < nstatesdo; j++) {
//            uint64_t p = i * nstatesup + j;
//            double m = 0;
//            m += mdia[p] * c[p];
//            q[p] += m;
//        }
//    }
//    time2 = timeInSec();
//    printf("%f\n", time2 - time1);
#endif
    printf("MatVec Diagonal: ");
    time1 = timeInSec();
#pragma omp parallel for
    for (uint64_t i = 0; i < nstatesup; i++) {
        for (uint64_t j = 0; j < nstatesdo; j++) {
            uint64_t p = i * nstatesup + j;
            double m = 0;
            m += mdia[p] * c[p];
            q[p] += m;
        }
    }
    time2 = timeInSec();
    printf("%f\n", time2 - time1);
}

//void QPlusHTimesC4(double* &q, double* &c) {
//    uint64_t time1, time2;
//
//    printf("MatVec Hopping Down: ");
//    time1 = timeInSec();
//#pragma omp parallel for
//    for (uint64_t i = 0; i < nstatesup; i++) {
//        for (uint64_t j = 0; j < nstatesdo; j++) {
//            uint64_t p = i * nstatesup + j;
//            double m = 0;
//            for (uint64_t l = 0; l < nstatesdo; l++) {
//                uint64_t n = i * nstatesup + l;
//                m += mdo[j][l] * c[n];
//            }
//            q[p] += m;
//        }
//    }
//    time2 = timeInSec();
//    printf("%f\n", time2 - time1);
//
//    printf("MatVec Hopping Up: ");
//    time1 = timeInSec();
//    for (uint64_t kk = 0; kk < nstatesup; kk += blockup) {
//#pragma omp parallel for
//        for (uint64_t i = 0; i < nstatesup; i++) {
//            for (uint64_t j = 0; j < nstatesdo; j++) {
//                uint64_t p = i * nstatesup + j;
//                double m = 0;
//                uint64_t ub = fmin(kk + blockup, nstatesup);
//                for (uint64_t k = kk; k < ub; k++) {
//                    uint64_t n = k * nstatesup + j;
//                    m += mup[i][k] * c[n];
//                }
//                q[p] += m;
//            }
//        }
//    }
//
//    time2 = timeInSec();
//    printf("%f\n", time2 - time1);
//
//    printf("MatVec Diagonal: ");
//    time1 = timeInSec();
//#pragma omp parallel for
//    for (uint64_t i = 0; i < nstatesup; i++) {
//        for (uint64_t j = 0; j < nstatesdo; j++) {
//            uint64_t p = i * nstatesup + j;
//            double m = 0;
//            m += mdia[p] * c[p];
//            q[p] += m;
//        }
//    }
//    time2 = timeInSec();
//    printf("%f\n", time2 - time1);
//}

// solvetridiag
// Solves the alpha/beta tridiagonal matrix using gsl
// 
// gse: double value
// gsv: double value array of length nIterations
// alpha: double value array of length nIterations
// beta: double value array of length nIterations
// nIterations: number of iterations up to this point
// 
// There is much to be optimized around this function, but since 
// this is only of dimension nIterations, we don't care.

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
        gsl_vector_free(eval);
        gsl_matrix_free(evec);
        delete(d);

}


void lanczos2() {
    int j;
    int nIterations = 0;
    double* c;
    double* q;
    double dnew;
    double* evec;
    double dold = 10e+300;
    double* beta = new double[maxiter];
    double* alpha = new double[maxiter];
    c = new double[nstates];
    q = new double[nstates];
    double time1, time2;
    
#pragma omp parallel for
    for (uint64_t i = 0; i < nstates; i++) {
        c[i] = 1.0 / sqrt((double) nstates);
        q[i] = 0;
    }
    beta[0] = 0;
    j = 0;

    for (int n = 0; n < maxiter - 1; n++) {
        time1 = timeInSec();

        if (j != 0) {
#pragma omp parallel for
            for (uint64_t i = 0; i < nstates; i++) {
                double t = c[i];
                c[i] = q[i] / beta[j];
                q[i] = -beta[j] * t;
            }
        }
        QPlusHTimesC3(q, c);
        j++;

        double tmp = 0;
#pragma omp parallel for reduction(+:tmp)
        for (uint64_t i = 0; i < nstates; i++) {
            tmp += c[i] * q[i];
        }
        alpha[j] = tmp;
#pragma omp parallel for
        for (uint64_t i = 0; i < nstates; i++) {
            q[i] = q[i] - alpha[j] * c[i];
        }
        tmp = 0;
#pragma omp parallel for reduction(+:tmp)
        for (uint64_t i = 0; i < nstates; i++) {
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
        printf("Iteration time in sec %f\n", (time2 - time1));

    }
    delete(c);
    delete(q);
    delete(alpha);
    delete(beta);
    //    return 0;
}



//void lanczos() {
//    int j;
//    int nIterations = 0;
//    double* c;
//    double* q;
//    double* d;
//    double dold = 10000000;
//    double* beta = new double[maxiter];
//    double* alpha = new double[maxiter];
//    c = new double[nstates];
//    q = new double[nstates];
//    double time1, time2;
//    
//#pragma omp parallel for
//    for (uint64_t i = 0; i < nstates; i++) {
//        c[i] = 1.0 / sqrt((double) nstates);
//        q[i] = 0;
//    }
//    beta[0] = 0;
//    j = 0;
//
//    for (int n = 0; n < maxiter - 1; n++) {
//        time1 = timeInSec();
//
//        if (j != 0) {
//#pragma omp parallel for
//            for (uint64_t i = 0; i < nstates; i++) {
//                double t = c[i];
//                c[i] = q[i] / beta[j];
//                q[i] = -beta[j] * t;
//            }
//        }
//        QPlusHTimesC3(q, c);
//        j++;
//
//        double tmp = 0;
//#pragma omp parallel for reduction(+:tmp)
//        for (uint64_t i = 0; i < nstates; i++) {
//            tmp += c[i] * q[i];
//        }
//        alpha[j] = tmp;
//#pragma omp parallel for
//        for (uint64_t i = 0; i < nstates; i++) {
//            q[i] = q[i] - alpha[j] * c[i];
//        }
//        tmp = 0;
//#pragma omp parallel for reduction(+:tmp)
//        for (uint64_t i = 0; i < nstates; i++) {
//            tmp += q[i] * q[i];
//        }
//        beta[j] = sqrt(tmp);
//        nIterations = j;
//        printf("beta[%d]=%e\n", j, beta[j]);
//
//        // ------------ SolverTriDiagonal begin
//        gsl_matrix *mm = gsl_matrix_alloc(nIterations, nIterations);
//
//        for (int i = 0; i < nIterations; i++) {
//            for (int j = 0; j < nIterations; j++) {
//                gsl_matrix_set(mm, i, j, 0);
//            }
//        }
//
//        for (int i = 0; i < nIterations; i++) {
//            gsl_matrix_set(mm, i, i, alpha[i]);
//        }
//
//        for (int i = 1; i < nIterations; i++) {
//            gsl_matrix_set(mm, i - 1, i, beta[i - 1]);
//            gsl_matrix_set(mm, i, i - 1, beta[i - 1]);
//        }
//
//        gsl_vector *eval = gsl_vector_alloc(nIterations);
//        gsl_matrix *evec = gsl_matrix_alloc(nIterations, nIterations);
//
//        gsl_eigen_symmv_workspace *w =
//                gsl_eigen_symmv_alloc(nIterations);
//
//        gsl_eigen_symmv(mm, eval, evec, w);
//        gsl_eigen_symmv_free(w);
//        gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);
//        
//
//        double* d = new double[nIterations];
//        for (int i = 0; i < nIterations; i++)
//            d[i] = gsl_vector_get(eval, i);
//        gsl_matrix_free(mm);
//
//        printf("Eigenvalue = %g\n", d[0]);
//        // ------------ SolverTriDiagonal end
//
//        printf("Difference %e - %e = %e\n", d[0], dold, fabs(d[0] - dold));
//        if (j > 5) {
//            if ((fabs(d[0] - dold) < convCrit)) {
//                break;
//            }
//        }
//        dold = d[0];
//        delete(d);
//        time2 = timeInSec();
//        printf("Iteration time in sec %f\n", (time2 - time1));
//
//    }
//    delete(c);
//    delete(q);
//    delete(alpha);
//    delete(beta);
//    //    return 0;
//}

void setupMatrix() {
    double time1, time2;
    time1 = timeInSec();
#ifdef _SPARSE_

#pragma omp parallel
    {
        double* sparse_vbuffer = new double[sparse_buffersize];
        uint64_t* sparse_cbuffer = new uint64_t[sparse_buffersize];
#pragma omp for
        for (uint64_t i = 0; i < nstatesup; i++) {
            int n = 0;
            for (uint64_t k = 0; k < nstatesup; k++) {
                double m = matrixelementT(sup[i], sup[k]);
//                printf("%f ", m);
                if (m != 0.0e+0) {                   
                    sparse_cbuffer[n] = k;
//                    sparse_cbuffer[n] = i * nstatesup + k;
                    sparse_vbuffer[n] = m;
//                    printf("%d %d %d %f \n",i,k,n,m);
                    n++;
                }
            }
//            printf("\n");
            sparse_nup[i] = n;
            sparse_pcup[i] = new uint64_t[n];
            sparse_pvup[i] = new double[n];
            for (int r = 0; r < n; r++) {
                sparse_pcup[i][r] = sparse_cbuffer[r];
                sparse_pvup[i][r] = sparse_vbuffer[r];
//                printf("%d %d %f \n",i,sparse_cbuffer[r],sparse_vbuffer[r]);
//                printf("%d %d %f \n",i,sparse_pcup[i][r],sparse_pvup[i][r]);
//                printf("\n");
            }
        }

#pragma omp for
        for (uint64_t j = 0; j < nstatesdo; j++) {
//            double* sparse_vbuffer = new double[sparse_buffersize];
//            uint64_t* sparse_cbuffer = new uint64_t[sparse_buffersize];
            int n = 0;
            for (uint64_t l = 0; l < nstatesdo; l++) {
                double m = matrixelementT(sdo[j], sdo[l]);
                if (m != 0.0e+0) {
//                    sparse_cbuffer[n] = j*nstatesup+l;
                    sparse_cbuffer[n] = l;
                    sparse_vbuffer[n] = m;
                    n++;
                }
            }
            sparse_ndo[j] = n;
            sparse_pcdo[j] = new uint64_t[n];
            sparse_pvdo[j] = new double[n];
            for (int r = 0; r < n; r++) {
                sparse_pcdo[j][r] = sparse_cbuffer[r];
                sparse_pvdo[j][r] = sparse_vbuffer[r];
            }
        }
#pragma omp for
        for (uint64_t i = 0; i < nstatesup; i++) {
            for (uint64_t j = 0; j < nstatesdo; j++) {
//                int p = i * nstatesup + j;
                uint64_t p = i * nstatesdo + j;
                mdia[p] = matrixelementU(sup[i], sdo[j]);
                mdia[p] += matrixelementMu(sup[i], sdo[j]);
            }
        }
    }
#else
#pragma omp parallel for
    for (uint64_t i = 0; i < nstatesup; i++) {
        for (uint64_t k = 0; k < nstatesup; k++) {
            mup[i][k] = matrixelementT(sup[i], sup[k]);
//            if(mup[i][k]!=0.0e+0){
//                printState(sup[i],nsites);
//                printf(" ");
//                printState(sup[k],nsites);
//                printf(" ");
//                printState(sup[i]^sup[k],nsites);
//                printf(" %f\n",mup[i][k]);
//            }
        }
    }
#pragma omp parallel for
    for (uint64_t j = 0; j < nstatesdo; j++) {
        for (uint64_t l = 0; l < nstatesdo; l++) {
            mdo[j][l] = matrixelementT(sdo[j], sdo[l]);
//            if(mdo[j][l]!=0.0e+0){
//                printf("%ld %ld %f\n",j,l,mdo[j][l]);
//            }
        }
    }
#pragma omp parallel for
    for (uint64_t i = 0; i < nstatesup; i++) {
        for (uint64_t j = 0; j < nstatesdo; j++) {
            uint64_t p = i * nstatesup + j;
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
    
#endif
    time2 = timeInSec();
    printf("Setup: %f\n", time2 - time1);

}

void init() {
    nstatesup = nStatesPerSpin(nsites, neup);
    nstatesdo = nStatesPerSpin(nsites, nedo);
    nstates = nstatesup*nstatesdo;
    sup = new uint64_t[nstatesup];
    sdo = new uint64_t[nstatesdo];
#ifdef _SPARSE_
    sparse_nup = new uint32_t[nstatesup];
    sparse_ndo = new uint32_t[nstatesdo];

    sparse_pvup = new double*[nstatesup];
    sparse_pvdo = new double*[nstatesdo];

    sparse_pcup = new uint64_t*[nstatesup];
    sparse_pcdo = new uint64_t*[nstatesdo];
#else
    mup = new double*[nstatesup];
    mdo = new double*[nstatesdo];
#endif
    
    mdia = new double[nstates];

    setupBasis(sup, nsites, neup);
    setupBasis(sdo, nsites, nedo);

#ifdef _SPARSE_
// nothing to do here  
#else
    for (int i = 0; i < nstatesup; i++) {
        mup[i] = new double[nstatesup];
    }
    for (int i = 0; i < nstatesdo; i++) {
        mdo[i] = new double[nstatesdo];
    }
#endif
    setupMatrix();
}
#ifdef _SPARSE_
void printsparsematrix(){
    for(int i=0;i<nstatesup;i++){
        for(int n=0;n<sparse_nup[i];n++){
            printf("%d %d %f \n",i,sparse_pcup[i][n],sparse_pvup[i][n]);
        } 
    }

}
#endif
/*
 *
 */
int main(int argc, char** argv) {

    readConfigFromFile(argv[1]);
    printConfig();
    printf("blockup = %ld\n", blockup);
    init();
//    printsparsematrix();
    lanczos2();
    return (EXIT_SUCCESS);
}


