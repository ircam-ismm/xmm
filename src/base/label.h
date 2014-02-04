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
    
protected:
    int intLabel;
    string symLabel;
};

#endif
