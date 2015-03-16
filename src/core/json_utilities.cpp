/*
 * json_utilities.cpp
 *
 * Set of utility functions for JSON I/O
 *
 * Contact:
 * - Jules Françoise <jules.francoise@ircam.fr>
 *
 * This code has been initially authored by Jules Françoise
 * <http://julesfrancoise.com> during his PhD thesis, supervised by Frédéric
 * Bevilacqua <href="http://frederic-bevilacqua.net>, in the Sound Music
 * Movement Interaction team <http://ismm.ircam.fr> of the
 * STMS Lab - IRCAM, CNRS, UPMC (2011-2015).
 *
 * Copyright (C) 2015 UPMC, Ircam-Centre Pompidou.
 *
 * This File is part of XMM.
 *
 * XMM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * XMM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with XMM.  If not, see <http://www.gnu.org/licenses/>.
 */

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