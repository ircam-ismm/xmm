//
// label.h
//
// Simple Label class (int & symbolic)
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#ifndef __mhmm__label__
#define __mhmm__label__

#include <iostream>
#include "json_utilities.h"

using namespace std;

/**
 * @ingroup TrainingSet
 * @class Label
 * @brief Label of a data phrase
 * @details Possible types are int and string
 */
class Label {
public:
    /**
     * @brief Type of the Label: 'INT' or 'SYM'
     */
    enum {INT, SYM} type;
    
    /**
     * @brief Default constructor
     * @details The default label type is INT, with value 0
     */
    Label();
    
    /**
     * @brief Constructor.
     * @details The default label type is INT, with value 0
     * @param l integer label
     */
    explicit Label(int l);

    /**
     * @brief Constructor from C++ std::string
     * @param l symbolic label
     */
    explicit Label(string l);

    /**
     * @brief Constructor from C-like string
     * @param l symbolic label
     */
    explicit Label(char* l);

    /**
     * @brief Copy Constructor
     */
    Label(Label const& src);
    
    /**
     * @brief Assignment
     * @param src source Label
     */
    Label& operator=(Label const& src);

    /**
     * @brief Assignment from int
     * @param l integer label
     */
    Label& operator=(int l);

    /**
     * @brief Assignment from std::string
     * @param l symbolic label
     */
    Label& operator=(string l);
    
    /**
     * @brief Assignment from C-like string
     * @param l symbolic label as C-string
     */
    Label& operator=(char* l);
    
    /**
     * @brief Check label equality
     * @param src source Label
     * @return true if labels are equal (type and value)
     */
    bool operator==(Label const& src) const;

    /**
     * @brief Check label inequality
     * @param src source Label
     * @return true if labels are different (type or value)
     */
    bool operator!=(Label const& src) const;

    /**
     * @brief Check label inequality
     * @param src source Label
     * @return true if the label is inferior to source (alphabetical order is used for symbolic labels)
     */
    bool operator<(Label const& src) const;

    /**
     * @brief Check label inequality
     * @param src source Label
     * @return true if the label is inferior or equal to source (alphabetical order is used for symbolic labels)
     */
    bool operator<=(Label const& src) const;

    /**
     * @brief Check label inequality
     * @param src source Label
     * @return true if the label is superior to source (alphabetical order is used for symbolic labels)
     */
    bool operator>(Label const& src) const;

    /**
     * @brief Check label inequality
     * @param src source Label
     * @return true if the label is superior or equal to source (alphabetical order is used for symbolic labels)
     */
    bool operator>=(Label const& src) const;
    
    /**
     * Get integer label value
     * @throw runtime_error if label type is not INT
     * @return integer label
     */
    int getInt() const;
    
    /**
     * Get symbolic label value
     * @throw runtime_error if label type is not SYM
     * @return symbolic label
     */
    string getSym() const;
    
    /**
     * Set integer label value => sets label type to INT
     * @param l integer label
     */
    void setInt(int l);
    
    /**
     * Try to set an integer from a string that contains one.
     * @param l integer label stored in a string
     * @return true if the integer label could be set
     */
    bool trySetInt(string l);
    
    /**
     * Set symbolic label value => sets label type to SYM
     * @param l symbolic label
     */
    void setSym(string l);
    
    /**
     * Set symbolic label value => sets label type to SYM
     * @param l symbolic label as C-string
     */
    void setSym(char* l);
    
    /**
     * @brief Write to JSON Node
     * @return JSON Node containing the label information
     */
    JSONNode to_json() const;
    
    /**
     * @brief Read from JSON Node
     * @param root JSON Node containing the label information
     */
    void from_json(JSONNode root);
    
    /**
     * @brief print label as c++ std::string
     */
    string as_string() const;
    
    /**
     * @brief Insertion operator
     * @param stream output stream
     * @param l label
     */
    friend ostream& operator<<(std::ostream& stream, Label const& l);
    
protected:
    /**
     * @brief Integer value
     * @details [long description]
     */
    int intLabel_;

    /**
     * @brief symbolic value
     */
    string symLabel_;
};

/**
 * @brief Check if the string contains an integer
 * @param s std::string to check
 * @return true if the string contains an integer
 */
bool is_number(const string& s);

/**
 * @brief Get integer from string
 */
int to_int(const string& s);

#endif
