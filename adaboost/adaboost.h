#ifndef ADABOOST_LIBRARY_H
#define ADABOOST_LIBRARY_H
#include <iostream>
#include <vector>
#include <limits>
#include <math.h>
using namespace std;


struct Decision_Function
{
    int feature_index = -1;
    double error = INT_MAX;
    double threshold=INT_MIN;
    // vector<double> predictions;

};

struct Decision_Stump
{
    Decision_Function decision_function;
    double alpha_t;

};

double update_weights(vector<int> &labels, double curr_error, vector<double> & weights,vector<double> & feature_vals,double threshold)
{
    double alpha_t = 0.5 * log((1-curr_error)/(curr_error));

    double weights_sum = 0;

    cout<< "\n Alpha_t "<<alpha_t<<"\n";

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
    return alpha_t;
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

Decision_Function get_feature_threshold_curr_feature(vector<int> &labels, vector<double> &weights, vector<double> &feature_vals,vector<double> &feature_thresholds)
{
    Decision_Function best_feature_threshold;

    for(int i=0;i<feature_thresholds.size();++i)
    {
        double curr_error = get_error_curr_feature_val(labels,weights,feature_vals,feature_thresholds[i]);
        if (curr_error <= best_feature_threshold.error)
        {
            best_feature_threshold.error = curr_error;
            best_feature_threshold.threshold=feature_thresholds[i];
            best_feature_threshold.feature_index = -1;

        }
    }
    return best_feature_threshold;
}

Decision_Stump get_best_feature_stump(vector<int> &labels, vector<vector<double> > &feature_vals,vector<double> &weights)
{
    Decision_Stump best_feature_stump;
    for(int i=0;i<feature_vals.size();++i)
    {
        Decision_Function curr_decision_function = get_feature_threshold_curr_feature(labels,weights,feature_vals[i],feature_vals[i]);
        if (curr_decision_function.error < best_feature_stump.decision_function.error)
        {
            best_feature_stump.decision_function.error = curr_decision_function.error;
            best_feature_stump.decision_function.threshold = curr_decision_function.threshold;
            best_feature_stump.decision_function.feature_index = i;
        }
    }
    if (best_feature_stump.decision_function.error >=0.0005)
        best_feature_stump.alpha_t = update_weights(labels,best_feature_stump.decision_function.error,weights,
                       feature_vals[best_feature_stump.decision_function.feature_index],best_feature_stump.decision_function.threshold);

    return best_feature_stump;

}

vector<Decision_Stump> fit_weak_classifers(vector<int> &labels, vector<vector<double> > &ds_t,int t)
{

    vector<Decision_Stump> weak_classifiers;

    vector<double>weights(labels.size(),1.0/sizeof(ds_t));
    for(int i=0;i<t;++i) {
        Decision_Stump curr_decision_stump = get_best_feature_stump(labels, ds_t,weights);
        weak_classifiers.push_back(curr_decision_stump);
        if (curr_decision_stump.decision_function.error <=0.00005)
            return weak_classifiers;
    }
    return weak_classifiers;

}


class Adaoost{

    vector<Decision_Stump> weak_classifiers;
    public:

        void fit(vector<int> &labels, vector<vector<double> > &ds_t,int t)
        {
            weak_classifiers = fit_weak_classifers(labels,ds_t,t);
            for(int i =0;i<weak_classifiers.size();++i)
            {
                cout<<"Weak Classifier ["<<i<<"] index" << weak_classifiers[i].decision_function.feature_index<<" "\
                     <<weak_classifiers[i].decision_function.threshold<<" ";
                cout<<"Alpha_t "<<weak_classifiers[i].alpha_t<< "\n" ;
            }
        }
        vector<int> predict(vector<vector<double> > &X) {
            vector<int> pred_labels(X.size(),1);

            for (int i = 0; i < X.size(); ++i) {
                double negative_label_weight = 0;
                double positive_label_weight = 0;
                for (int j = 0; j < weak_classifiers.size(); ++j) {

                    Decision_Stump curr_classifier = weak_classifiers[j];
                    double  curr_feat_val = X[i][curr_classifier.decision_function.feature_index];
                    if ( curr_feat_val <= curr_classifier.decision_function.threshold) {
                        negative_label_weight += curr_classifier.alpha_t;
                    } else {
                        positive_label_weight += curr_classifier.alpha_t;
                    }

                }

                if (negative_label_weight >= positive_label_weight) {
                    pred_labels[i]=-1;
                } else {
                    pred_labels[i]=1;
                }
            }
            return pred_labels;
        }


};


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