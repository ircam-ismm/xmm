//
// listener.h
//
// Common Definitions for Machine Learning Models
//
// Copyright (C) 2014 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise <jules.francoise@ircam.fr>
//

#ifndef mhmm_listener_object_h
#define mhmm_listener_object_h

#include "json_utilities.h"
#include <fstream>

using namespace std;

const vector<float> NULLVEC_FLOAT;
const vector<double> NULLVEC_DOUBLE;

typedef unsigned int rtml_flags;

/**
 * @brief Flags for the construction of data phrases and models
 */
enum FLAGS {
    /**
     * @brief no specific Flag: Unimodal and own Memory
     */
    NONE = 0,
    
    /**
     * @brief Defines a shared memory phrase.
     * @details If this flag is used, data can only be passed by pointer.
     * Recording functions are disabled and the memory cannot be freed from a Phrase Object.
     */
    SHARED_MEMORY = 1 << 1,
    
    /**
     * @brief Defines is a phrase is used to store bimodal data
     * @details If this falg is used, the phrase contains 2 arrays for input and output modalities
     */
    BIMODAL = 1 << 2,
    
    /**
     * @brief Defines is the model is used by a hierarchical Algorithm
     * @details If this falg is used, the phrase contains 2 arrays for input and output modalities
     */
    HIERARCHICAL = 1 << 3
};

/**
 * @defgroup Utilities Utilities
 */

/**
 * @ingroup Utilities
 * @class Listener
 * @brief Dummy class for handling training set notifications
 * @details It is an abstract class that contains a pure virtual method "notify" called by a training set
 * to notify changes of the training data
 */
class Listener {
public:
    friend class TrainingSet;
    
    /**
     * @brief pure virtual method for handling training set notifications.
     * @param attribute name of the modified attribute of the training set
     */
    virtual void notify(string attribute) = 0;
};

class Writable {
public:
    /**
     * @brief Write to JSON Node
     * @return JSON Node containing phrase information
     */
    virtual JSONNode to_json() const = 0;
    
    /**
     * @brief Read from JSON Node
     * @param root JSON Node containing phrase information
     * @throws JSONException if the JSON Node has a wrong format
     */
    virtual void from_json(JSONNode root) = 0;

#ifdef SWIGPYTHON
    /**
     * @brief write method for python wrapping ('write' keyword forbidden, name has to be different)
     */
    void writeFile(char* fileName)
    {
        ofstream outStream;
        outStream.open(fileName);
        JSONNode jsonfile = this->to_json();
        outStream << jsonfile.write_formatted();
        outStream.close();
    }
    
    /**
     * @brief read method for python wrapping ('read' keyword forbidden, name has to be different)
     */
    void readFile(char* fileName)
    {
        string jsonstring;
        ifstream inStream;
        inStream.open(fileName);
        inStream.seekg(0, ios::end);
        jsonstring.reserve(inStream.tellg());
        inStream.seekg(0, ios::beg);
        
        jsonstring.assign((istreambuf_iterator<char>(inStream)),
                          istreambuf_iterator<char>());
        JSONNode jsonfile = libjson::parse(jsonstring);
        this->from_json(jsonfile);
        
        inStream.close();
    }
    
    /**
     * @brief "print" method for python => returns the results of write method
     */
    char *__str__() {
        stringstream ss;
        JSONNode jsonfile = this->to_json();
        ss << jsonfile.write_formatted();
        string tmp = ss.str();
        char* cstr = strdup(tmp.c_str());
        return cstr;
    }
#endif
};

#endif
