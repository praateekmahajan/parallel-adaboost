#include "adaboost.h"

#include <iostream>
#include <vector>

using namespace std;

vector<vector<double> > transpose(vector<vector<double> > ds)
{
    vector<vector<double> > tr (ds[0].size(),vector<double>(ds.size(),0));
    for (int i = 0; i < ds.size(); ++i) {
        for (int j = 0; j < ds[i].size(); ++j) {
            tr[j][i]=ds[i][j];
        }
    }
    return tr;

}

int main() {
    //cout << "Hello, World!" ;

    int ds_len = 5;
    int ds_feat = 1;
    int t=4;

    vector<vector<double> > X(ds_len, vector<double>(ds_feat, 0));
    vector<int> labels(ds_len,-1);

    X[0][0]=0.3; labels[0] = 1;
    X[1][0]=0.4; labels[1] = -1;
    X[2][0]=0.5; labels[2] = -1;
    X[3][0]=0.6; labels[3] = 1;
    X[4][0]=0.7; labels[4] = 1;

//    X[5][0]=1.3; labels[5] = -1;
//    X[6][0]=1.4; labels[6] = -1;
//    X[7][0]=1.5; labels[7] = -1;
//    X[8][0]=1.6; labels[8] = 1;
//    X[9][0]=1.7; labels[9] = 1;


    for (int i = 0; i < X.size(); ++i) {
        for (int j = 0; j < X[i].size(); ++j) {
            cout << X[i][j] << " ";
        }
        cout<<" Label "<<labels[i];
        cout<<"\n";
    }

    vector<vector<double> > ds_t = transpose(X);
    Adaoost clf;
    clf.fit(labels,ds_t,t);
    vector<int> predictions = clf.predict(X);
    cout<<"\n";
    for(int i=0;i<predictions.size();++i)
    {
        cout<<"True Label "<<labels[i]<<" Predicted Label "<<predictions[i]<<"\n";

    }


    return 0;
}