//
// label.h
//
// Simple Label class (int & symbolic)
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#ifndef __mhmm__label__
#define __mhmm__label__

#include <iostream>
#include "libjson.h"

using namespace std;

/*!
 @class Label
 @brief Label of a data phrase
 Possible types are int and string
 */
class Label {
public:
    enum {INT, SYM} type;
    
    /*!
     Constructor. Default label type is INT. Default Value is 0
     */
    Label();
    Label(int l);
    Label(string l);
    Label(char* l);
    
    Label& operator=(Label const& src);
    Label& operator=(int l);
    Label& operator=(string l);
    Label& operator=(char* l);
    
    bool operator==(Label const& src) const;
    bool operator!=(Label const& src) const;
    bool operator<(Label const& src) const;
    bool operator<=(Label const& src) const;
    bool operator>(Label const& src) const;
    bool operator>=(Label const& src) const;
    
    /*!
     Get integer label value
     @throw RTMLException if label type is not INT
     */
    int getInt() const;
    
    /*!
     Get symbolic label value
     @throw RTMLException if label type is not SYM
     */
    string getSym() const;
    
    /*!
     Set integer label value => sets label type to INT
     */
    void setInt(int l);
    
    /*!
     Set symbolic label value => sets label type to SYM
     */
    void setSym(string l);
    
    /*!
     Set symbolic label value => sets label type to SYM
     */
    void setSym(char* l);
    
    /*!
     Write to JSON Node
     */
    virtual JSONNode to_json() const ;
    
    /*!
     Read from JSON Node
     */
    virtual void from_json(JSONNode root);
    
    /*!
     print label as c++ string
     */
    string as_string();
    
protected:
    int intLabel;
    string symLabel;
};

ostream& operator<<(std::ostream& stream, Label const& l);

#endif
