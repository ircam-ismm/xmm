//
// label.cpp
//
// Simple Label class (int & symbolic)
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#include "label.h"
#include <sstream>

Label::Label() : type(INT), intLabel_(0), symLabel_("")
{}

Label::Label(int l) : type(INT), intLabel_(l), symLabel_("")
{}

Label::Label(string l) : type(SYM), intLabel_(0), symLabel_(l)
{}

Label::Label(char* l) : type(SYM), intLabel_(0), symLabel_(l)
{}

Label::Label(Label const& src) : type(src.type), intLabel_(src.intLabel_), symLabel_(src.symLabel_)
{}

Label& Label::operator=(Label const& src)
{
    if (this != &src) {
        this->type = src.type;
        this->intLabel_ = src.intLabel_;
        this->symLabel_ = src.symLabel_;
    }
    return *this;
}

Label& Label::operator=(int l)
{
    this->type = INT;
    this->intLabel_ = l;
    this->symLabel_ = "";
    return *this;
}

Label& Label::operator=(string l)
{
    this->type = SYM;
    this->intLabel_ = 0;
    this->symLabel_ = l;
    return *this;
}

Label& Label::operator=(char* l)
{
    this->type = SYM;
    this->intLabel_ = 0;
    this->symLabel_ = l;
    return *this;
}

bool Label::operator==(Label const& src) const
{
    if (!(this->type == src.type))
        return false;
    if (this->type == INT)
        return (this->intLabel_ == src.intLabel_);
    return (this->symLabel_ == src.symLabel_);
}

bool Label::operator!=(Label const& src) const
{
    return !operator==(src);
}

bool Label::operator<(Label const& src) const
{
    if (type == INT)
        return intLabel_ < src.intLabel_;
    else
        return symLabel_ < src.symLabel_;
}

bool Label::operator<=(Label const& src) const
{
    return (operator<(src) || operator==(src));
}

bool Label::operator>(Label const& src) const
{
    return !operator<=(src);
}

bool Label::operator>=(Label const& src) const
{
    return !operator<(src);
}

int Label::getInt() const
{
    if (type != INT)
        throw runtime_error("Can't get INT from SYM label");
    return intLabel_;
}

string Label::getSym() const
{
    if (type != SYM)
        throw runtime_error("Can't get SYM from INT label");
    return symLabel_;
}

void Label::setInt(int l)
{
    type = INT;
    intLabel_ = l;
}

bool Label::trySetInt(string l)
{
    if (is_number(l)) {
        setInt(to_int(l));
        return true;
    }
    return false;
}


void Label::setSym(string l)
{
    type = SYM;
    symLabel_ = l;
}

void Label::setSym(char* l)
{
    type = SYM;
    symLabel_ = l;
}

JSONNode Label::to_json() const 
{
    JSONNode json_label(JSON_NODE);
	json_label.set_name("label");
    if (type == INT) {
        json_label.push_back(JSONNode("type", "INT"));
        json_label.push_back(JSONNode("value", intLabel_));
    } else {
        json_label.push_back(JSONNode("type", "SYM"));
        json_label.push_back(JSONNode("value", symLabel_));
    }
	return json_label;
}

void Label::from_json(JSONNode root)
{
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root.name());
        JSONNode::const_iterator root_it = root.begin();
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "type")
            throw JSONException("Wrong name: was expecting 'type'", root_it->name());
        if (root_it->type() != JSON_STRING)
            throw JSONException("Wrong type: was expecting 'JSON_STRING'", root_it->name());
        if (root_it->as_string() == "INT")
            type = INT;
        else
            type = SYM;
        ++root_it;
        
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "value")
            throw JSONException("Wrong name: was expecting 'value'", root_it->name());
        if (type == INT) {
            if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
            intLabel_ = root_it->as_int();
        } else {
            if (root_it->type() != JSON_STRING)
                throw JSONException("Wrong type: was expecting 'JSON_STRING'", root_it->name());
            symLabel_ = root_it->as_string();
        }
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}

string Label::as_string() const
{
    stringstream ss;
    ss << *this;
    return ss.str();
}

ostream& operator<<(std::ostream& stream, Label const& l)
{
    if (l.type == Label::INT)
        stream << l.getInt();
    else
        stream << l.getSym();
    return stream;
}

bool is_number(const string& s)
{
    string::const_iterator it = s.begin();
    while (it != s.end() && isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

int to_int(const string& s)
{
    istringstream myString(s);
    int value;
    myString >> value;
    return value;
}
