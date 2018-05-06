#include "adaboost.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

using namespace std;


int main() {
    //cout << "Hello, World!" ;


    int t=5;

    // vector<vector<double> > X(ds_len, vector<double>(ds_feat, 0));
    //vector<int> labels(ds_len,-1);

    vector<vector<double> > X;
    vector<int> labels;
    // The data file, without labels and first row.
    ifstream data_file("mnist_data.csv");
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
    ifstream label_file("mnist_label.csv");


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

    Adaoost clf;
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
    cout<<"Accuracy is "<<acc/predictions.size();
    cout<<"\n";


    return 0;
}