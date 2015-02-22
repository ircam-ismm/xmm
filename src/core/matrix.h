//
// matrix.h
//
// Matrix utilities
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#ifndef __mhmm__matrix__
#define __mhmm__matrix__

#include <iostream>
#include <vector>
#include <exception>
#include <stdexcept>
#include <cmath>

using namespace std;

/**
 * @brief Epsilon value for Matrix inversion
 * @details  defines 
 */
const double EPS = 1.0e-9;

/**
 * @ingroup Utility
 * @class Matrix
 * @brief Dirty and very incomplete Matrix Class
 * @details Contains few utilities for matrix operations, with possibility to share data with vectors
 * @tparam numType data type of the matrix (should be used with float/double)
 */
template <typename numType>
class Matrix {
public:
    /**
     * @brief Vector iterator
     */
    typedef typename vector<numType>::iterator iterator;
    
    /**
     * @brief number of rows of the matrix
     */
    int nrows;

    /**
     * @brief number of columns of the matrix
     */
    int ncols;

    /**
     * @brief Matrix Data if not shared
     */
    vector<numType> _data;

    /**
     * @brief Data iterator
     * @details Can point to own data vector, or can be shared with another container.
     */
    iterator data;

    /**
     * @brief Defines if the matrix has its own data
     */
    bool ownData;
    
    /**
     * @brief Default Constructor
     * @param ownData defines if the matrix stores the data itself (true by default)
     */
    Matrix(bool ownData=true);

    /**
     * @brief Square Matrix Constructor
     * @param nrows Number of rows (defines a square matrix)
     * @param ownData defines if the matrix stores the data itself (true by default)
     */
    Matrix(int nrows, bool ownData=true);

    /**
     * @brief Constructor
     * @param nrows Number of rows
     * @param ncols Number of columns
     * @param ownData defines if the matrix stores the data itself (true by default)
     */
    Matrix(int nrows, int ncols, bool ownData=true);

    /**
     * @brief Constructor from vector (shared data)
     * @param nrows Number of rows
     * @param ncols Number of columns
     * @param data_it iterator to the vector data
     */
    Matrix(int nrows, int ncols, iterator data_it);

    /**
     * @brief Destructor
     * @details Frees memory only if data is owned
     */
    ~Matrix();
    
    /**
     * @brief Resize the matrix
     * @param nrows Number of rows
     * @param ncols Number of columns
     * @throws runtime_error if the matrix is not square
     */
    void resize(int nrows, int ncols);


    /**
     * @brief Resize a Square Matrix
     * @param nrows Number of rows
     */
    void resize(int nrows);

    /**
     * @brief Compute the Sum of the matrix
     * @return sum of all elements in the matrix
     */
    float sum();

    /**
     * @brief Print the matrix
     */
    void print();
    
    /**
     * @brief Compute the transpose matrix
     * @return pointer to the transpose Matrix
     * @warning Memory is allocated for the new matrix (need to be freed)
     */
    Matrix<numType>* transpose();

    /**
     * @brief Compute the product of matrices
     * @return pointer to the Matrix resulting of the product
     * @warning Memory is allocated for the new matrix (need to be freed)
     * @throws runtime_error if the matrices have wrong dimensions
     */
    Matrix<numType>* product(Matrix const* mat);

    /**
     * @brief Compute the Pseudo-Inverse of a Matrix
     * @param det Determinant (computed with the inversion)
     * @return pointer to the inverse Matrix
     * @warning Memory is allocated for the new matrix (need to be freed)
     */
    Matrix<numType>* pinv(double *det);

    /**
     * @brief Compute the Gauss-Jordan Inverse of a Square Matrix
     * @param det Determinant (computed with the inversion)
     * @return pointer to the inverse Matrix
     * @warning Memory is allocated for the new matrix (need to be freed)
     * @throws runtime_error if the matrix is not square
     * @throws runtime_error if the matrix is not invertible
     */
    Matrix<numType>* gauss_jordan_inverse(double *det) const;

    /**
     * @brief Swap 2 lines of the matrix
     * @param i index of the first line
     * @param j index of the second line
     */
    void swap_lines(int i, int j);

    /**
     * @brief Swap 2 columns of the matrix
     * @param i index of the first column
     * @param j index of the second column
     */
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
        throw runtime_error("Gauss-Jordan inversion: Can't invert Non-square matrix");
    }
    *det = 1.0f;
    Matrix<numType> mat(nrows, ncols*2);
    Matrix<numType> new_mat(nrows, ncols*2);
    
    int n = nrows; 
    
    // Create matrix
    for (int i=0 ; i<n ; i++) {
        for (int j=0; j<n; j++) {
            mat._data[i*2*n+j] = data[i*n+j];
        }
        mat._data[i*2*n+n+i] = 1;
    }
    
    for (int k=0; k<n; k++) {
        int i(k);
        while (abs(mat._data[i*2*n+k]) < EPS) {
            i++;
            if (i==n) {
                throw runtime_error("Non-invertible matrix");
            }
        }
        *det *= mat._data[i*2*n+k];
        
        // if found > Exchange lines
        if (i != k) {
            mat.swap_lines(i, k);
        }
        
        new_mat._data = mat._data;
        
        for (int j=0; j<2*n; j++) {
            new_mat._data[k*2*n+j] /= mat._data[k*2*n+k];
        }
        for (i=0; i<n; i++) {
            if (i != k) {
                for (int j=0; j<2*n; j++) {
                    new_mat._data[i*2*n+j] -= mat._data[i*2*n+k] * new_mat._data[k*2*n+j];
                }
            }
        }
        mat._data = new_mat._data;
    }
    
    Matrix<numType> *dst = new Matrix<numType>(nrows, ncols);
    for (int i=0 ; i<n ; i++)
        for (int j=0; j<n; j++)
            dst->_data[i*n+j] = mat._data[i*2*n+n+j];
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

#endif