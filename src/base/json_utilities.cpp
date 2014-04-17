//
// json_utilities.cpp
//
// Set of utility functions for JSON I/O
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#include "json_utilities.h"

template <>
void json2array(JSONNode root, float* a, int n)
{
    if (root.type() != JSON_ARRAY)
        throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root.name());
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw JSONException("JSON 2 Array: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw JSONException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_float();
    }
}

template <>
void json2array(JSONNode root, double* a, int n)
{
    if (root.type() != JSON_ARRAY)
        throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root.name());
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw JSONException("JSON 2 Array: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw JSONException("JSON 2 Vector: Wrong type");
        a[i++] = double(array_it->as_float());
    }
}

template <>
void json2array(JSONNode root, bool* a, int n)
{
    if (root.type() != JSON_ARRAY)
        throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root.name());
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw JSONException("JSON 2 Array: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw JSONException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_bool();
    }
}

template <>
void json2array(JSONNode root, string* a, int n)
{
    // Get Dimensions
    if (root.type() != JSON_ARRAY)
        throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root.name());
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw JSONException("JSON 2 Array: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw JSONException("JSON 2 Vector: Wrong type");
        a[i++] = libjson::to_std_string(array_it->as_string());
    }
}

template <>
void json2vector(JSONNode root, vector<float>& a, int n)
{
    // Get Dimensions
    if (root.type() != JSON_ARRAY)
        throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root.name());
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw JSONException("JSON 2 Vector: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw JSONException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_float();
    }
}

template <>
void json2vector(JSONNode root, vector<double>& a, int n)
{
    // Get Dimensions
    if (root.type() != JSON_ARRAY)
        throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root.name());
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw JSONException("JSON 2 Vector: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw JSONException("JSON 2 Vector: Wrong type");
        a[i++] = double(array_it->as_float());
    }
}

template <>
void json2vector(JSONNode root, vector<bool>& a, int n)
{
    // Get Dimensions
    if (root.type() != JSON_ARRAY)
        throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root.name());
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw JSONException("JSON 2 Vector: Index out of bounds");
        if (array_it->type() != JSON_BOOL)
            throw JSONException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_bool();
    }
}

template <>
void json2vector(JSONNode root, vector<string>& a, int n)
{
    // Get Dimensions
    if (root.type() != JSON_ARRAY)
        throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root.name());
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw JSONException("JSON 2 Vector: Index out of bounds");
        if (array_it->type() != JSON_STRING)
            throw JSONException("JSON 2 Vector: Wrong type");
        a[i++] = libjson::to_std_string(array_it->as_string());
    }
}