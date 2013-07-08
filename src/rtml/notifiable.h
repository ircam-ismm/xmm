//
//  notifiable_object.h
//  mhmm
//
//  Created by Jules Francoise on 28/01/13.
//
//

#ifndef mhmm_notifiable_object_h
#define mhmm_notifiable_object_h

#include <string>
#include <ostream>
#include <fstream>

using namespace std;

/*!
 @class Notifiable
 Dummy class for handling training set notifications\n
 It is an abstract class that contains a pure virtual method "notify" called by a training set 
 to notify changes of the training data\n
 (also includes read/write pure virtual methods)
 */
class Notifiable {
public:
    /*!
     pure virtual method for handling training set notifications.
     @param attribute name of the modified attribute of the training set
     */
    virtual void notify(string attribute) = 0;
    
    /*!
     pure virtual method for file IO => writing
     @param outStream output stream
     @param writeTrainingSet defines if training set must be saved with the object
     */
    virtual void write(ostream& outStream, bool writeTrainingSet=false) = 0;
    
    /*!
     pure virtual method for file IO => reading
     @param inStream input stream
     @param readTrainingSet defines if training set must be loaded with the object
     */
	virtual void read(istream& inStream, bool readTrainingSet=false) = 0;
    
#ifdef SWIGPYTHON
    /*!
     write method for python wrapping (name has to be different)
     */
    void writeFile(char* fileName, bool writeTrainingSet=false)
    {
        ofstream outStream;
		outStream.open(fileName);
		this->write(outStream, writeTrainingSet);
		outStream.close();
    }
    
    /*!
     read method for python wrapping (name has to be different)
     */
    void readFile(char* fileName, bool readTrainingSet=false)
    {
        ifstream inStream;
		inStream.open(fileName);
		this->read(inStream, readTrainingSet);
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
