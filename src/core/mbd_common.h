/*
 * mbd_common.h
 *
 * Common Definitions for XMM
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

#ifndef mhmm_listener_object_h
#define mhmm_listener_object_h

#include "json_utilities.h"
#include <fstream>

using namespace std;

const vector<float> NULLVEC_FLOAT;
const vector<double> NULLVEC_DOUBLE;

typedef unsigned int rtml_flags;

/**
 * @defgroup Core Core Classes and Distributions
 */

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
 * @brief Abstract class for handling training set notifications
 * @details It is an abstract class that contains a pure virtual method "notify" called by a training set
 * to notify changes of the training data
 */
class Listener {
public:
    friend class TrainingSet;
    
    virtual ~Listener() {}
    
    /**
     * @brief pure virtual method for handling training set notifications.
     * @param attribute name of the modified attribute of the training set
     */
    virtual void notify(string attribute) = 0;
};

/**
 * @ingroup Utilities
 * @brief Abstract class for handling JSON + File I/O
 * @details the JSON I/O methods need to be implemented. writeFile and readFile methods
 * can be used in Python for file I/O. The __str__() Python method is implemented to use
 * with "print" in Python. It return the pretty-printed JSON String.
 */
class Writable {
public:
    virtual ~Writable() {}
    
    /*@{*/
    /** @name JSON I/O */
    /**
     * @brief Write to JSON Node
     * @return JSON Node containing phrase information
     * @todo include type attribute in each to_json function
     */
    virtual JSONNode to_json() const = 0;
    
    /**
     * @brief Read from JSON Node
     * @param root JSON Node containing phrase information
     * @throws JSONException if the JSON Node has a wrong format
     * @todo add force_conversion optional argument?
     */
    virtual void from_json(JSONNode root) = 0;
    
    /*@}*/
    
#ifdef SWIGPYTHON
    /*@{*/
    /** @name Python File I/O (#ifdef SWIGPYTHON) */
    /**
     * @brief write method for python wrapping ('write' keyword forbidden, name has to be different)
     * @warning only defined if SWIGPYTHON is defined
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
     * @warning only defined if SWIGPYTHON is defined
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
     * @warning only defined if SWIGPYTHON is defined
     */
    string __str__() {
        stringstream ss;
        JSONNode jsonfile = this->to_json();
        ss << jsonfile.write_formatted();
        string tmp = ss.str();
        return tmp;
    }
    /*@}*/
#endif
};

#endif
