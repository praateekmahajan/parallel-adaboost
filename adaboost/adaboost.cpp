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

    int ds_len = 10;
    int ds_feat = 2;
    int t=2;

    vector<vector<double> > ds(ds_len, vector<double>(ds_feat, 0));
    vector<int> labels(ds_len,-1);

    ds[0][0]=0.3; labels[0] = 1;
    ds[1][0]=0.3; labels[1] = 1;
    ds[2][0]=0.3; labels[2] = 1;
    ds[3][0]=0.4; labels[3] = 1;
    ds[4][0]=0.5; labels[4] = 1;

    ds[5][0]=1.3; labels[5] = 1;
    ds[6][0]=1.4; labels[6] = -1;
    ds[7][0]=1.5; labels[7] = -1;
    ds[8][0]=1.6; labels[8] = -1;
    ds[9][0]=1.7; labels[9] = -1;


    for (int i = 0; i < ds.size(); ++i) {
        for (int j = 0; j < ds[i].size(); ++j) {
            cout << ds[i][j] << " ";
        }
        cout<<" Label "<<labels[i];
        cout<<"\n";
    }
    vector<double>weights(ds_len,1.0/ds_len);

    // Take the transpose of curr ds to make computation down the road easier

    vector<vector<double> > ds_t = transpose(ds);

    for(int i=0;i<t;++i) {
        cout<<"\n Loop = "<<i<<"\n";

        Feature_Stump best_feature_stump = get_best_feature_stump(labels, weights, ds_t);

        cout << "\nBest Feature Index " << best_feature_stump.feature_index;
        cout << "\nBest Feature Threshold " << best_feature_stump.threshold;
        cout << "\nBest Feature Error " << best_feature_stump.error;

    }
//    for(int i=0;i<weights.size();++i)
//    {
//        cout<<weights[i]<<"\n";
//    }
    cout<<"\n\n";


    return 0;
}