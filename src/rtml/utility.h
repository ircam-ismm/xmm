//
//  utility.h
//  rtml
//
//  Created by Jules Francoise on 20/01/13.
//
//

#ifndef rtml_utility_h
#define rtml_utility_h

#include <iostream>
#include <vector>
#include "rtmlexception.h"

#define EPSILON_GAUSSIAN 1.0e-40

using namespace std;

#pragma mark -
#pragma mark Memory Allocation

template <typename T>
T* reallocate(T *src, int dim_src, int dim_dst) {
    T *dst = new T[dim_dst];
    
    if (!src) return dst;
    
    if (dim_dst > dim_src) {
        memcpy(dst, src, dim_src*sizeof(T));
    } else {
        memcpy(dst, src, dim_dst*sizeof(T));
    }
    delete[] src;
    return dst;
}

#pragma mark -
#pragma mark Gaussian Distribution
double gaussianProbabilityFullCovariance(const float *obs, std::vector<float>::iterator mean,
                                         double covarianceDeterminant,
                                         std::vector<float>::iterator inverseCovariance,
                                         int dimension);
double gaussianProbabilityFullCovariance_GestureSound(const float *obs_gesture,
                                                      const float *obs_sound,
                                                      std::vector<float>::iterator mean,
                                                      double covarianceDeterminant,
                                                      std::vector<float>::iterator inverseCovariance,
                                                      int dimension_gesture,
                                                      int dimension_sound);

#pragma mark -
#pragma mark Vector Utilies
void vectorCopy(vector<float>::iterator dst_it, vector<float>::iterator src_it, int size);
void vectorCopy(vector<double>::iterator dst_it, vector<double>::iterator src_it, int size);
void vectorMultiply(std::vector<float>::iterator dst_it, std::vector<float>::iterator src_it, int size);
void vectorMultiply(std::vector<double>::iterator dst_it, std::vector<double>::iterator src_it, int size);

#pragma mark -
#pragma mark File IO
const int MAX_STR_SIZE(4096);

void skipComments(istream *s);

#pragma mark -
#pragma mark Simple Ring Buffer
template <typename T, int channels>
class RingBuffer {
public:
	// Constructor
	RingBuffer(unsigned int length_ = 1)
    {
        length = length_;
        for (int c=0; c<channels; c++) {
            data[c].resize(length);
        }
        index = 0;
        full = false;
    }
    
	~RingBuffer()
    {
        for (int c=0; c<channels; c++) {
            data[c].clear();
        }
    }
    
    T operator()(int c, int i)
    {
        if (c >= channels)
            throw RTMLException("channel out of bounds", __FILE__, __FUNCTION__, __LINE__);
        int m = full ? length : index;
        if (i >= m)
            throw RTMLException("index out of bounds", __FILE__, __FUNCTION__, __LINE__);
        return data[c][i];
    }
	
	// methods
	void clear()
    {
        index = 0;
        full = false;
    }
    
    void push(T const value)
    {
        if (channels > 1)
            throw RTMLException(" you must pass a vector or array", __FILE__, __FUNCTION__, __LINE__);
        data[0][index] = value;
        index++;
        if (index == length)
            full = true;
        index %= length;
    }
    
	void push(T const *value)
    {
        for (int c=0; c<channels; c++)
        {
            data[c][index] = value[c];
        }
        index++;
        if (index == length)
            full = true;
        index %= length;
    }
    
    void push(vector<T> const &value)
    {
        for (int c=0; c<channels; c++)
        {
            data[c][index] = value[c];
        }
        index++;
        if (index == length)
            full = true;
        index %= length;
    }
    
	unsigned int size() const
    {
        return length;
    }
    
    unsigned int size_t() const
    {
        return (full ? length : index);
    }
    
	void resize(unsigned int length_)
    {
        if (length_ == length) return;
        if (length_ > length) {
            full = false;
        } else if (index >= length_) {
            full = true;
            index = 0;
        }
        length = length_;
        for (int c=0; c<channels; c++) {
            data[c].resize(length);
        }
    }
    
    vector<T> mean() const
    {
        vector<T> _mean(channels, 0.0);
        int size = full ? length : index;
        for (int c=0; c<channels; c++)
        {
            for (int i=0; i<size; i++) {
                _mean[c] += data[c][i];
            }
            _mean[c] /= T(size);
        }
        return _mean;
    }
	
protected:
    vector<T> data[channels];
	unsigned int length;
	unsigned int index;
	bool full;
};

#endif
