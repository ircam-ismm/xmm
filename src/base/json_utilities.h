//
//  json_utilities.h
//  mhmm
//
//  Created by Jules Francoise on 14/02/2014.
//
//

#ifndef mhmm_json_utilities_h
#define mhmm_json_utilities_h

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
            throw RTMLException("JSON 2 Array: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw RTMLException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_int();
    }
}

template <>
void json2array(JSONNode root, float* a, int n)
{
    // Get Dimensions
    assert(root.type() == JSON_ARRAY);
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw RTMLException("JSON 2 Array: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw RTMLException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_float();
    }
}

template <>
void json2array(JSONNode root, double* a, int n)
{
    // Get Dimensions
    assert(root.type() == JSON_ARRAY);
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw RTMLException("JSON 2 Array: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw RTMLException("JSON 2 Vector: Wrong type");
        a[i++] = double(array_it->as_float());
    }
}

template <>
void json2array(JSONNode root, bool* a, int n)
{
    // Get Dimensions
    assert(root.type() == JSON_ARRAY);
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw RTMLException("JSON 2 Array: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw RTMLException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_bool();
    }
}

template <>
void json2array(JSONNode root, string* a, int n)
{
    // Get Dimensions
    assert(root.type() == JSON_ARRAY);
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw RTMLException("JSON 2 Array: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw RTMLException("JSON 2 Vector: Wrong type");
        a[i++] = libjson::to_std_string(array_it->as_string());
    }
}

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
            throw RTMLException("JSON 2 Vector: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw RTMLException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_int();
    }
}

template <>
void json2vector(JSONNode root, vector<float>& a, int n)
{
    // Get Dimensions
    assert(root.type() == JSON_ARRAY);
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw RTMLException("JSON 2 Vector: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw RTMLException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_float();
    }
}

template <>
void json2vector(JSONNode root, vector<double>& a, int n)
{
    // Get Dimensions
    assert(root.type() == JSON_ARRAY);
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw RTMLException("JSON 2 Vector: Index out of bounds");
        if (array_it->type() != JSON_NUMBER)
            throw RTMLException("JSON 2 Vector: Wrong type");
        a[i++] = double(array_it->as_float());
    }
}

template <>
void json2vector(JSONNode root, vector<bool>& a, int n)
{
    // Get Dimensions
    assert(root.type() == JSON_ARRAY);
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw RTMLException("JSON 2 Vector: Index out of bounds");
        if (array_it->type() != JSON_BOOL)
            throw RTMLException("JSON 2 Vector: Wrong type");
        a[i++] = array_it->as_bool();
    }
}

template <>
void json2vector(JSONNode root, vector<string>& a, int n)
{
    // Get Dimensions
    assert(root.type() == JSON_ARRAY);
    unsigned int i = 0;
    for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
    {
        if (i >= n)
            throw RTMLException("JSON 2 Vector: Index out of bounds");
        if (array_it->type() != JSON_STRING)
            throw RTMLException("JSON 2 Vector: Wrong type");
        a[i++] = libjson::to_std_string(array_it->as_string());
    }
}


#endif
