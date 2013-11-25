//
//  mhmm_utility.cpp
//  mhmm
//
//  Created by Jules Francoise on 17/10/12.
//
//

#include "utility.h"
#include <cmath>
#include <exception>
#include <stdexcept>

using namespace std;

#pragma mark -
#pragma mark Gaussian Distribution
double gaussianProbabilityFullCovariance(const float *obs,
                                         vector<float>::iterator mean,
                                         double covarianceDeterminant,
                                         vector<float>::iterator inverseCovariance,
                                         int dimension)
{
    if (covarianceDeterminant == 0.0) throw runtime_error("Covariance Matrix is not invertible");
    
    double euclidianDistance(0.0);
    double tmp(0.0);
    for (int l=0; l<dimension; l++) {
        tmp = 0.0;
        for (int k=0; k<dimension; k++) {
            tmp += inverseCovariance[l*dimension+k] * (obs[k] - mean[k]);
        }
        euclidianDistance += (obs[l] - mean[l]) * tmp;
    }
    
    double p = exp(-0.5 * euclidianDistance)* EPSILON_GAUSSIAN / sqrt(covarianceDeterminant * pow(2*M_PI, (double)dimension));
    
    if(p < 1e-80 || isnan(p) || isinf(abs(p))) p = 1e-80;
    
    return p;
}

double gaussianProbabilityFullCovariance_GestureSound(const float *obs_gesture,
                                                      const float *obs_sound,
                                                      std::vector<float>::iterator mean,
                                                      double covarianceDeterminant,
                                                      std::vector<float>::iterator inverseCovariance,
                                                      int dimension_gesture,
                                                      int dimension_sound)
{
    if (covarianceDeterminant == 0.0) throw runtime_error("Covariance Matrix is not invertible");
    
    int dimension_total = dimension_gesture + dimension_sound;
    double euclidianDistance(0.0);
    double tmp(0.0);
    for (int l=0; l<dimension_total; l++) {
        tmp = 0.0;
        for (int k=0; k<dimension_gesture; k++) {
            tmp += inverseCovariance[l*dimension_total+k] * (obs_gesture[k] - mean[k]);
        }
        for (int k=dimension_gesture; k<dimension_total; k++) {
            tmp += inverseCovariance[l*dimension_total+k] * (obs_sound[k-dimension_gesture] - mean[k]);
        }
        euclidianDistance += ( ((l<dimension_gesture) ? obs_gesture[l] : obs_sound[l-dimension_gesture]) - mean[l]) * tmp;
    }
    
    double p = exp(-0.5 * euclidianDistance) * EPSILON_GAUSSIAN / sqrt(covarianceDeterminant * pow(2*M_PI, (double)dimension_total));
    
    // if(p < 1e-80 || isnan(p) || isinf(abs(p))) p = 1e-80;
    
    return p;
}


#pragma mark -
#pragma mark Vector utilities
//void vectorCopy(vector<float>::iterator dst_it, vector<float>::iterator src_it, int size)
//{
//    for (int i=0; i<size; i++) {
//        dst_it[i] = src_it[i];
//    }
//}
//
//void vectorCopy(vector<double>::iterator dst_it, vector<double>::iterator src_it, int size)
//{
//    for (int i=0; i<size; i++) {
//        dst_it[i] = src_it[i];
//    }
//}
//
//void vectorMultiply(std::vector<float>::iterator dst_it, std::vector<float>::iterator src_it, int size)
//{
//    for (int i=0; i<size; i++) {
//        dst_it[i] *= src_it[i];
//    }
//}

void vectorMultiply(std::vector<double>::iterator dst_it, std::vector<double>::iterator src_it, int size)
{
    for (int i=0; i<size; i++) {
        dst_it[i] *= src_it[i];
    }
}

#pragma mark -
#pragma mark File IO
void skipComments(istream *s)
{
    streampos prevPos;
    char tmp_str[MAX_STR_SIZE];
    do {
        prevPos = s->tellg();
        s->getline(tmp_str, MAX_STR_SIZE);
    } while (*tmp_str == '#' || *tmp_str == 0 || *tmp_str == ' ' || *tmp_str == '\n');
    s->seekg(prevPos);
}