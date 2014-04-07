//
// listener.h
//
// Base Class for objects receiveing notifications
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#ifndef mhmm_listener_object_h
#define mhmm_listener_object_h

#include <string>
#include <ostream>
#include <fstream>
#include "libjson.h"
#include "json_utilities.h"

using namespace std;

/*!
 @class Listener
 Dummy class for handling training set notifications\n
 It is an abstract class that contains a pure virtual method "notify" called by a training set
 to notify changes of the training data\n
 (also includes read/write pure virtual methods)
 */
class Listener {
public:
    /*!
     pure virtual method for handling training set notifications.
     @param attribute name of the modified attribute of the training set
     */
    virtual void notify(string attribute) = 0;
    
    /*!
     pure virtual method for file IO => writing
     @param outStream output stream
     */
    virtual void write(ostream& outStream) = 0;
    
    /*!
     pure virtual method for file IO => reading
     @param inStream input stream
     */
    virtual void read(istream& inStream) = 0;
    
    /*!
     Write to JSON Node
     */
    virtual JSONNode to_json() const = 0;
    
    /*!
     Read from JSON Node
     */
    virtual void from_json(JSONNode root) = 0;
    
    /*
    void writeFile(string filename)
    {
        JSONNode root = this->to_json();
        ofstream outputFile;
        outputFile.open(filename);
        outputFile.close();
    }
    
    void readFile(string filename)
    {
        ifstream inputFile;
    }
    //*/
    
#ifdef SWIGPYTHON
    /*!
     write method for python wrapping ('write' keyword forbidden, name has to be different)
     */
    void writeFile(char* fileName)
    {
        ofstream outStream;
        outStream.open(fileName);
        JSONNode jsonfile = this->to_json();
        outStream << jsonfile.write_formatted();
        outStream.close();
    }
    
    /*!
     read method for python wrapping ('read' keyword forbidden, name has to be different)
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
        cout << jsonstring << endl;
        
        inStream.close();
    }
    
    /*!
     "print" method for python => returns the results of write method
     */
    char *__str__() {
        stringstream ss;
        write(ss);
        string tmp = ss.str();
        char* cstr = strdup(tmp.c_str());
        return cstr;
    }
#endif
};

#endif
