//
// matrix.h
//
// Matrix utilities
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#ifndef __mhmm__matrix__
#define __mhmm__matrix__

#include <iostream>
#include <vector>
#include <exception>
#include <stdexcept>
#include <cmath>

using namespace std;

namespace momos {
    const double EPS = 1.0e-9;
    /*!
     * @class Matrix
     * @brief Dirty and very incomplete Matrix Library
     *
     * Contains few utilities for matrix operations, with possibility to share data with vectors
     */
#pragma mark -
#pragma mark Class definition
    template <typename numType>
    class Matrix {
    public:
        typedef typename vector<numType>::iterator iterator;
        
        int nrows;
        int ncols;
        vector<numType> _data;
        iterator data;
        bool ownData;
        
        Matrix(bool ownData_=true);
        Matrix(int nrows_, bool ownData_=true);
        Matrix(int nrows_, int ncols_, bool ownData_=true);
        Matrix(int nrows_, int ncols_, iterator data_it);
        ~Matrix();
        
        void resize(int nrows_, int ncols_);
        void resize(int nrows_);
        float sum();
        void print();
        
        Matrix<numType>* transpose();
        Matrix<numType>* product(Matrix const* mat);
        Matrix<numType>* pinv(double *det);
        Matrix<numType>* gauss_jordan_inverse(double *det) const;
        void swap_lines(int i, int j);
        void swap_columns(int i, int j);
    };
    
#pragma mark -
#pragma mark Constructors
    template <typename numType>
    Matrix<numType>::Matrix(bool ownData_) : nrows(0), ncols(0), ownData(ownData_) {}
    
    template <typename numType>
    Matrix<numType>::Matrix(int nrows_, bool ownData_)
    {
        nrows = nrows_;
        ncols = nrows_;
        ownData = ownData_;
        if (ownData) {
            _data.assign(nrows*ncols, numType(0.0));
            data = _data.begin();
        }
    }
    
    template <typename numType>
    Matrix<numType>::Matrix(int nrows_, int ncols_, bool ownData_)
    {
        nrows = nrows_;
        ncols = ncols_;
        ownData = ownData_;
        if (ownData) {
            _data.assign(nrows*ncols, numType(0.0));
            data = _data.begin();
        }
    }
    
    template <typename numType>
    Matrix<numType>::Matrix(int nrows_, int ncols_, iterator data_)
    {
        nrows = nrows_;
        ncols = ncols_;
        ownData = false;
        data = data_;
    }
    
    template <typename numType>
    Matrix<numType>::~Matrix()
    {
        if (ownData)
            _data.clear();
    }
    
#pragma mark -
#pragma mark Utilities
    template <typename numType>
    void Matrix<numType>::resize(int nrows_)
    {
        if (nrows != ncols)
            throw runtime_error("Matrix is not square");
        
        nrows = nrows_;
        ncols = nrows;
        _data.resize(nrows*ncols);
    }
    
    template <typename numType>
    void Matrix<numType>::resize(int nrows_, int ncols_)
    {
        nrows = nrows_;
        ncols = ncols_;
        _data.resize(nrows*ncols);
    }
    
    template <typename numType>
    void Matrix<numType>::print()
    {
        for (int i=0 ; i<nrows ; i++) {
            for (int j=0 ; j<ncols ; j++) {
                cout << data[i*ncols+j] << " ";
            }
            cout << endl;
        }
    }
    
    template <typename numType>
    float Matrix<numType>::sum()
    {
        float sum_(0.);
        for (int i=0; i<nrows*ncols; i++)
            sum_ += data[i];
        return sum_;
    }
    
#pragma mark -
#pragma mark Basic Operations
    template <typename numType>
    Matrix<numType>* Matrix<numType>::transpose()
    {
        Matrix<numType> *out = new Matrix<numType>(ncols, nrows);
        for (int i=0 ; i<ncols ; i++) {
            for (int j=0 ; j<nrows ; j++) {
                out->data[i*nrows+j] = data[j*ncols+i];
            }
        }
        return out;
    }
    
    template <typename numType>
    Matrix<numType>* Matrix<numType>::product(Matrix const* mat)
    {
        if (ncols != mat->nrows)
            throw runtime_error("Wrong dimensions for matrix product");
        
        Matrix<numType> *out = new Matrix<numType>(nrows, mat->ncols);
        for (int i=0 ; i<nrows ; i++) {
            for (int j=0 ; j<mat->ncols ; j++) {
                out->data[i*mat->ncols+j] = 0.;
                for (int k=0 ; k<ncols ; k++) {
                    out->data[i*mat->ncols+j] += data[i*ncols+k] * mat->data[k*mat->ncols+j];
                }
            }
        }
        return out;
    }
    
#pragma mark -
#pragma mark Pseudo-inverse
    template <typename numType>
    Matrix<numType>* Matrix<numType>::pinv(double* det)
    {
        Matrix<numType> *inverse = NULL;
        if (nrows == ncols) {
            inverse = gauss_jordan_inverse(det);
            if (inverse) {
                return inverse;
            }
        }
        
        inverse = new Matrix<numType>(ncols, nrows);
        Matrix<numType> *transp, *prod, *dst;
        transp = this->transpose();
        if (nrows >= ncols) {
            prod = transp->product(this);
            dst = prod->gauss_jordan_inverse(det);
            inverse = dst->product(transp);
        } else {
            prod = this->product(transp);
            dst = prod->gauss_jordan_inverse(det);
            inverse = transp->product(dst);
        }
        *det = 0;
        delete transp;
        delete prod;
        delete dst;
        return inverse;
    }
    
    template <typename numType>
    Matrix<numType>* Matrix<numType>::gauss_jordan_inverse(double *det) const
    {
        if (nrows != ncols) {
            throw runtime_error("Gauss-Jordan: Can't invert Non-quare matrix");
        }
        *det = 1.0f;
        Matrix<numType> mat(nrows, ncols*2);
        Matrix<numType> new_mat(nrows, ncols*2);
        
        int n = nrows;
        
        // Create matrix
        for (int i=0 ; i<n ; i++) {
            for (int j=0; j<n; j++) {
                mat.data[i*2*n+j] = data[i*n+j];
            }
            mat.data[i*2*n+n+i] = 1;
        }
        
        for (int k=0; k<n; k++) {
            int i(k);
            while (abs(mat.data[i*2*n+k]) < EPS) {
                i++;
                if (i==n) {
                    throw runtime_error("Error: non-invertible matrix");
                }
            }
            *det *= mat.data[i*2*n+k];
            
            // if found > Exchange lines
            if (i != k) {
                mat.swap_lines(i, k);
            }
            
            new_mat._data = mat._data;
            
            for (int j=0; j<2*n; j++) {
                new_mat.data[k*2*n+j] /= mat.data[k*2*n+k];
            }
            for (i=0; i<n; i++) {
                if (i != k) {
                    for (int j=0; j<2*n; j++) {
                        new_mat.data[i*2*n+j] -= mat.data[i*2*n+k] * new_mat.data[k*2*n+j];
                    }
                }
            }
            mat._data = new_mat._data;
        }
        
        Matrix<numType> *dst = new Matrix<numType>(nrows, ncols);
        for (int i=0 ; i<n ; i++)
            for (int j=0; j<n; j++)
                dst->data[i*n+j] = mat.data[i*2*n+n+j];
        return dst;
    }
    
    template <typename numType>
    void Matrix<numType>::swap_lines(int i, int j)
    {
        numType tmp;
        for (int k=0; k<ncols; k++) {
            tmp = data[i*ncols+k];
            data[i*ncols+k] = data[j*ncols+k];
            data[j*ncols+k] = tmp;
        }
    }
    
    template <typename numType>
    void Matrix<numType>::swap_columns(int i, int j)
    {
        numType tmp;
        for (int k=0; k<nrows; k++) {
            tmp = data[k*ncols+i];
            data[k*ncols+i] = data[k*ncols+j];
            data[k*ncols+j] = tmp;
        }
    }
}

#endif