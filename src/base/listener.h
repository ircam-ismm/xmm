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
    
#ifdef SWIGPYTHON
    /*!
     write method for python wrapping ('write' keyword forbidden, name has to be different)
     */
    void writeFile(char* fileName)
    {
        ofstream outStream;
        outStream.open(fileName);
        this->write(outStream);
        outStream.close();
    }
    
    /*!
     read method for python wrapping ('read' keyword forbidden, name has to be different)
     */
    void readFile(char* fileName)
    {
        ifstream inStream;
        inStream.open(fileName);
        this->read(inStream);
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
