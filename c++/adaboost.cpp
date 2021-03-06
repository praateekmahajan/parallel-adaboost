#include "adaboost_best.h"
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <sstream>

using namespace std;

int timeval_subtract (struct timeval * result, struct timeval * x, struct timeval * y)
{
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



int main(int argc, char** argv) {

    int num_threads =1;
    if(argc!=0)
        num_threads = atoi(argv[1]);

    int t;
    if(argc>=3)
        t=atoi(argv[2]);
    else
        t = 2;

    string num_egs;
    string num_ft;
    if (argc == 5)
    {
        num_egs = argv[3];
        num_ft = argv[4];
    }
    else
    {
        num_egs = "100";
        num_ft = "20";	     
    }

    string FILENAME = "data/" + num_egs + "_" + num_ft;

    /* Variables for timing */
    struct timeval ta, tb, tresult;

    /* get initial time */

    omp_set_num_threads(num_threads);

    vector<vector<double> > X;
    vector<int> labels;
    // The data file, without labels and first row.
    string dataf = FILENAME + "_data.csv";
    ifstream data_file(dataf.c_str());
    string line;

    while (data_file.good()) {
        getline(data_file, line);
        std::vector<double_t > vect;
        double i;
        std::stringstream ss(line);
        while (ss >> i)
        {
            vect.push_back(i);

            if (ss.peek() == ',')
                ss.ignore();
        }

                X.push_back(vect);
    }
    X.pop_back();

    // Label file, expects, 1 and -1 as postive and negative labels
    string label = FILENAME + "_label.csv";
    ifstream label_file(label.c_str());


    while (label_file.good()) {

        getline(label_file, line);
        double i;
        std::stringstream ss(line);
        ss >> i;
        labels.push_back(i);
    }



    // Let's time only the predict since that is what we are focussing on!
    gettimeofday ( &ta, NULL );
    AdaBoost clf;
    clf.fit(X,labels,t);
    gettimeofday ( &tb, NULL );
    timeval_subtract ( &tresult, &tb, &ta );


    vector<int> predictions = clf.predict(X);
    double acc;
    for(int i=0;i<predictions.size();++i)
    {
        if(labels[i]==predictions[i])
        {
            acc++;
        }
    }
    cout<<"Accuracy "<<acc/predictions.size();
    cout<<"\n";

    /* get initial time */
    printf ("%s\t%s\t%d\t%d\t%lu\t%lu\n", num_egs.c_str(), num_ft.c_str(), num_threads, t, tresult.tv_sec, tresult.tv_usec);

    return 0;
}
