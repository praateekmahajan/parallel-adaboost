#include "adaboost.h"
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/time.h>

#include <sstream>

using namespace std;

FILENAME = "../data/temp";
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
    //cout << "Hello, World!" ;

//    cout << "You have entered " << argc
//         << " arguments:" << "\n";

    int t;

//    for (int i = 0; i < argc; ++i)
//        cout << argv[i] << "\n";

    if(argc==3)
        t=atoi(argv[2]);
    else
        t = 2;

    int num_threads =1;
    if(argc!=0)
        num_threads = atoi(argv[1]);

    /* Variables for timing */
    struct timeval ta, tb, tresult;

    /* get initial time */
    gettimeofday ( &ta, NULL );

    omp_set_num_threads(num_threads);



    // vector<vector<double> > X(ds_len, vector<double>(ds_feat, 0));
    //vector<int> labels(ds_len,-1);

    vector<vector<double> > X;
    vector<int> labels;
    // The data file, without labels and first row.
    ifstream data_file(FILENAME + "_data.csv");
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
    ifstream label_file(FILENAME + "_label.csv");


    while (label_file.good()) {

        getline(label_file, line);
        double i;
        std::stringstream ss(line);
        ss >> i;
        labels.push_back(i);
    }


    // etc.
//        std::vector<std::string> vec;
//        split(vec, line, is_any_of(delimeter));
//        dataList.push_back(vec);


//    X[0][0]=10; labels[0] = 1;
//    X[1][0]=30; labels[1] = 1;
//    X[2][0]=40; labels[2] = -1;
//    X[3][0]=60; labels[3] = -1;
//    X[4][0]=90; labels[4] = 1;
//
//    X[0][1]=10;
//    X[1][1]=30;
//    X[2][1]=40;
//    X[3][1]=60;
//    X[4][1]=90;
//
//    X[5][0]=1.3; labels[5] = -1;
//    X[6][0]=1.4; labels[6] = -1;
//    X[7][0]=1.5; labels[7] = -1;
//    X[8][0]=1.6; labels[8] = 1;
//    X[9][0]=1.7; labels[9] = -1;
//
//
//    cout<<X.size();
//    for (int i = 0; i < X.size(); i++) {
//        cout<<i<<" ";
//        for (int j = 0; j < X[i].size(); j++) {
//            cout << X[i][j] << " ";
//        }
//        cout<<" "<<labels[i]<<"\n";
//    }

    AdaBoost clf;
    clf.fit(X,labels,t);
    vector<int> predictions = clf.predict(X);
    cout<<"\n";
    double acc;
    for(int i=0;i<predictions.size();++i)
    {
        if(labels[i]==predictions[i])
        {
            acc++;
        }
    }
    cout<<"\nAccuracy "<<acc/predictions.size();
    cout<<"\n";

    /* get initial time */
    gettimeofday ( &tb, NULL );

    timeval_subtract ( &tresult, &tb, &ta );

    printf ("%lu\t%lu\t%d\n", tresult.tv_sec, tresult.tv_usec, num_threads );

    return 0;
}
