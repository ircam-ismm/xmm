//
// json_utilities.h
//
// Set of utility functions for JSON I/O
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#ifndef mhmm_json_utilities_h
#define mhmm_json_utilities_h

#include <cassert>
#include <iostream>
#include <string>
#include <exception>
#include <vector>
#include "libjson.h"

using namespace std;

/**
 * @ingroup Utilities
 * @class JSONException
 * @brief Simple Exception class for handling JSON I/O errors
 */
class JSONException : public exception
{
public:
    /**
     * @brief Default Constructor
     * @param message error message
     * @param nodename name of the JSON node where the error occurred
     */
    JSONException(string message="", string nodename="")
    : message_(message), nodename_(nodename)
    {}
    
    /**
     * @brief Constructor From exception message
     * @param src Source Exception
     * @param nodename name of the
     */
    explicit JSONException(exception const& src, string nodename = "")
    : message_(src.what()), nodename_(nodename)
    {}
    
    /**
     * @brief Copy Constructor
     * @param src Source JSON exception
     */
    JSONException(JSONException const& src)
    {
        this->_copy(this, src);
    }
    
    /**
     * @brief Assigment
     * @param src Source JSON exception
     */
    JSONException& operator=(JSONException const& src)
    {
        if(this != &src)
        {
            _copy(this, src);
        }
        return *this;
    }
    
    /**
     * @brief Copy between two JSON exceptions
     * @param dst destination JSON exception
     * @param src Source JSON exception
     */
    virtual void _copy(JSONException *dst,
                       JSONException const& src)
    
    {
        dst->message_ = src.message_;
        dst->nodename_ = src.nodename_;
    }
    
    /**
     * @brief Destructor
     */
    virtual ~JSONException() throw()
    {}
    
    /**
     * @brief Get exception message
     * @return exception message
     */
    virtual const char * what() const throw()
    {
        string fullmsg = "Error reading JSON, Node '" + nodename_ + "': " + message_;
        return fullmsg.c_str();
    }
    
private:
    string message_;
    string nodename_;
};

/*@{*/
/**
 * @name JSON conversion to/from arrays and vectors
 */
template <typename T>
JSONNode array2json(T const* a, int n, string name="array")
{
	JSONNode json_data(JSON_ARRAY);
	json_data.set_name(name);
	for (int i=0 ; i<n ; i++) {
		json_data.push_back(JSONNode("", a[i]));
	}
	return json_data;
}

template <typename T>
void json2array(JSONNode root, T* a, int n)
{
    // Get Dimensions
    assert(root.type() == JSON_ARRAY);
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw JSONException("JSON 2 Array: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw JSONException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_int();
    }
}

template <> void json2array(JSONNode root, float* a, int n);
template <> void json2array(JSONNode root, double* a, int n);
template <> void json2array(JSONNode root, bool* a, int n);
template <> void json2array(JSONNode root, string* a, int n);

template <typename T>
JSONNode vector2json(vector<T> const& a, string name="array")
{
	JSONNode json_data(JSON_ARRAY);
	json_data.set_name(name);
	for (size_t i=0 ; i<a.size() ; i++) {
		json_data.push_back(JSONNode("", a[i]));
	}
	return json_data;
}

template <typename T>
void json2vector(JSONNode root, vector<T>& a, int n)
{
    // Get Dimensions
    assert(root.type() == JSON_ARRAY);
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw JSONException("JSON 2 Vector: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw JSONException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_int();
    }
}

template <> void json2vector(JSONNode root, vector<float>& a, int n);
template <> void json2vector(JSONNode root, vector<double>& a, int n);
template <> void json2vector(JSONNode root, vector<bool>& a, int n);
template <> void json2vector(JSONNode root, vector<string>& a, int n);

/*@}*/

#endif
