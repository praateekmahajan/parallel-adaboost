#include <vector>
#include <string>
#include <sys/time.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <sstream>
#include <omp.h>
#include <assert.h>

using namespace std;

int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y) {
    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }

    /* Compute the time remaining to wait.
       tv_usec is certainly positive. */
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;

    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}

vector <vector<double>> transpose(vector <vector<double>> &ds) {
    //  Transposes the data set, done so that slicing feature wise is easier
    vector <vector<double>> tr(ds[0].size(), vector<double>(ds.size(), 0));
    int rows = ds.size();
    int cols = ds[0].size();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            tr[j][i] = ds[i][j];
        }
    }
    return tr;

}

vector <vector<double>> transpose1(vector <vector<double>> &ds) {
    //  Transposes the data set, done so that slicing feature wise is easier
    vector <vector<double>> tr(ds[0].size(), vector<double>(ds.size(), 0));
    int rows = ds.size();
    int cols = ds[0].size();
#pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            tr[j][i] = ds[i][j];
        }
    }
    return tr;

}


vector <vector<double>> transpose2(vector <vector<double>> &ds) {
    //  Transposes the data set, done so that slicing feature wise is easier
    vector <vector<double>> tr(ds[0].size(), vector<double>(ds.size(), 0));
    int i, j;
    int rows = ds.size();
    int cols = ds[0].size();

#pragma omp parallel for private(j)
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            tr[j][i] = ds[i][j];
        }
    }
    return tr;

}

vector <vector<double>> transpose3(vector <vector<double>> A) {
    //  Transposes the data set, done so that slicing feature wise is easier
    vector <vector<double>> B(A[0].size(), vector<double>(A.size(), 0));
    int i, j;
    int M = A.size();
    int N = A[0].size();
    int temp, p,q;
    for (i = 0; i < M; i += 18) {
        for (j = 0; j < N; j += 18) {
            for (p = i; (p < (i + 18)) && (p < M); p++) {
                for (q = j; (q < (j + 18)) && q < N; q++) {
                    if ((p) == (q)) {
// Only for diagonal elements, do we have the problem
                        temp = A[p][p];
                    } else {
                        B[p][q] = A[q][p];
                    }
                }
                if (i == j) {
/*Accessing B's set here is better because it reduces the
 *number of conflict misses
 */
                    B[p][p] = temp;
                }

            }

        }
    }


}


int main() {
    int rows = 20000;
    int cols = 10000;
    int i, j;
    struct timeval ta, tb, t1a, t1b, t2a, t2b, tresult, t1result, t2result;
    omp_set_num_threads(4);

    gettimeofday(&ta, NULL);
    vector <vector<double>> vec(rows, vector<double>(cols, 0));
    for (i = 0; i < vec.size(); i++) {
        for (j = 0; j < vec[0].size(); j++) {
            vec[i][j] = rand();
        }
    }
    gettimeofday(&tb, NULL);
    timeval_subtract(&tresult, &tb, &ta);
    printf("Flooding\t%lu.%lu\n", tresult.tv_sec, tresult.tv_usec);
//
//    /* Transpose 1 */
    gettimeofday(&t1a, NULL);
    vector <vector<double>> vec_t1 = transpose(vec);
    gettimeofday(&t1b, NULL);
    timeval_subtract(&t1result, &t1b, &t1a);
    printf("Transpose 1\t%lu.%lu\n", t1result.tv_sec, t1result.tv_usec);

    //    /* Transpose 2 */
    gettimeofday(&t1a, NULL);
    vector <vector<double>> vec_t2 = transpose3(vec);
    gettimeofday(&t1b, NULL);
    timeval_subtract(&t1result, &t1b, &t1a);
    printf("Transpose 2\t%lu.%lu\n", t1result.tv_sec, t1result.tv_usec);


    //    /* Transpose 2 */
    gettimeofday(&t1a, NULL);
    vector <vector<double>> vec_t3 = transpose2(vec);
    gettimeofday(&t1b, NULL);
    timeval_subtract(&t1result, &t1b, &t1a);
    printf("Transpose 3\t%lu.%lu\n", t1result.tv_sec, t1result.tv_usec);


    gettimeofday(&t2a, NULL);
    for (i = 0; i < vec.size(); i++) {
        for (j = 0; j < vec[0].size(); j++) {
            assert(vec_t2[j][i] == vec[i][j]);
            assert(vec_t3[j][i] == vec[i][j]);
        }
    }
    gettimeofday(&t2b, NULL);
    timeval_subtract(&t2result, &t2b, &t2a);
    printf("Checking\t%lu.%lu\n", t2result.tv_sec, t2result.tv_usec);


}
