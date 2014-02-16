//
// label.cpp
//
// Simple Label class (int & symbolic)
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#include "label.h"
#include "rtmlexception.h"

Label::Label() {
    type = INT;
    intLabel = 0;
    symLabel = "";
}

Label::Label(int l)
{
    type = INT;
    intLabel = l;
    symLabel = "";
}

Label::Label(string l)
{
    type = SYM;
    intLabel = 0;
    symLabel = l;
}

Label::Label(char* l)
{
    type = SYM;
    intLabel = 0;
    symLabel = l;
}

bool Label::operator==(Label const& src) const
{
    if (!this->type == src.type)
        return false;
    if (this->type == INT)
        return (this->intLabel == src.intLabel);
    return (this->symLabel == src.symLabel);
}

bool Label::operator!=(Label const& src) const
{
    return !operator==(src);
}

bool Label::operator<(Label const& src) const
{
    if (type == INT)
        return intLabel < src.intLabel;
    else
        return symLabel < src.symLabel;
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
        throw RTMLException("Can't get INT from SYM label");
    return intLabel;
}

string Label::getSym() const
{
    if (type != SYM)
        throw RTMLException("Can't get SYM from INT label");
    return symLabel;
}

void Label::setInt(int l) {
    type = INT;
    intLabel = l;
}

void Label::setSym(string l) {
    type = SYM;
    symLabel = l;
}

void Label::setSym(char* l) {
    type = SYM;
    symLabel = l;
}

JSONNode Label::to_json() const 
{
    JSONNode json_label(JSON_NODE);
	json_label.set_name("label");
    if (type == INT) {
        json_label.push_back(JSONNode("type", "INT"));
        json_label.push_back(JSONNode("value", intLabel));
    } else {
        json_label.push_back(JSONNode("type", "SYM"));
        json_label.push_back(JSONNode("value", symLabel));
    }
	return json_label;
}

void Label::from_json(JSONNode root)
{
    try {
        assert(root.type() == JSON_NODE);
        JSONNode::const_iterator root_it = root.begin();
        assert(root_it != root.end());
        assert(root_it->name() == "type");
        assert(root_it->type() == JSON_STRING);
        if (root_it->as_string() == "INT")
            type = INT;
        else
            type = SYM;
        root_it++;
        assert(root_it != root.end());
        assert(root_it->name() == "value");
        if (type == INT) {
            assert(root_it->type() == JSON_NUMBER);
            intLabel = root_it->as_int();
        } else {
            assert(root_it->type() == JSON_STRING);
            symLabel = root_it->as_string();
        }
    } catch (exception &e) {
        throw RTMLException("Error reading JSON, Node: " + root.name());
    }
}

string Label::as_string()
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