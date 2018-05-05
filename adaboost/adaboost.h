#ifndef ADABOOST_LIBRARY_H
#define ADABOOST_LIBRARY_H
#include <iostream>
#include <vector>
#include <limits>
#include <math.h>
using namespace std;


struct Feature_Stump
{
    int feature_index = -1;
    double error = INT_MAX;
    double threshold=INT_MIN;
    // vector<double> predictions;

};

void update_weights(vector<int> &labels, double curr_error, vector<double> & weights,vector<double> & feature_vals,double threshold)
{
    double alpha_t = 0.5 * log((1-curr_error)/(curr_error));

    double weights_sum = 0;

    // cout<< "Alpha_t"<<alpha_t;

    for(int i = 0;i<weights.size();++i)
    {
        if(feature_vals[i]<=threshold)
        {
            weights[i]=weights[i]* exp(alpha_t*labels[i]);
        }
        else
        {
            weights[i]=weights[i] * exp(-alpha_t*labels[i]);
        }
        weights_sum+=weights[i];
    }
    for(int i=0;i<weights.size();++i)
    {
        weights[i]=weights[i]/weights_sum;
    }

}


double get_error_curr_feature_val(vector<int> &labels,vector<double> & weights,vector<double> & feature_vals, double split_val)
{


    double curr_error = 0;

    for(int i=0;i<weights.size();++i)
    {
        if ( ((feature_vals[i]<=split_val) and (labels[i]!=-1)) or ((feature_vals[i]>split_val) and (labels[i]!=1)))
        {
            curr_error+=weights[i];
        }
    }

    return curr_error;

}

Feature_Stump get_feature_threshold_curr_feature(vector<int> &labels, vector<double> &weights, vector<double> &feature_vals,vector<double> &feature_thresholds)
{
    Feature_Stump best_feature_stump;
    for(int i=0;i<feature_thresholds.size();++i)
    {
        double curr_error = get_error_curr_feature_val(labels,weights,feature_vals,feature_thresholds[i]);
        if (curr_error <= best_feature_stump.error)
        {
            best_feature_stump.error = curr_error;
            best_feature_stump.threshold=feature_thresholds[i];
            best_feature_stump.feature_index = -1;

        }
    }
    return best_feature_stump;
}

Feature_Stump get_best_feature_stump(vector<int> &labels, vector<double> &weights, vector<vector<double> > &feature_vals)
{
    Feature_Stump best_feature_stump;
    for(int i=0;i<feature_vals.size();++i)
    {
        Feature_Stump curr_feature_stump = get_feature_threshold_curr_feature(labels,weights,feature_vals[i],feature_vals[i]);
        if (curr_feature_stump.error <= best_feature_stump.error)
        {
            best_feature_stump.error = curr_feature_stump.error;
            best_feature_stump.threshold = curr_feature_stump.threshold;
            best_feature_stump.feature_index = i;


        }
    }
    if (best_feature_stump.error >=0.0005)
        update_weights(labels,best_feature_stump.error,weights,feature_vals[best_feature_stump.feature_index],best_feature_stump.threshold);

    return best_feature_stump;

}




#endif



//def create_split(self, dataset, index, value, weight):
//left = []
//leftweight = []
//right = []
//rightweight = []
//for i, point in enumerate(dataset):
//if point[index] > value:
//right.append(point)
//rightweight.append(weight[i])
//else:
//left.append(point)
//leftweight.append(weight[i])
//return ((left, right), (leftweight, rightweight))