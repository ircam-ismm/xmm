/*
 * label.cpp
 *
 * Labels for classes (int + string)
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
    if (src.type == type) {
        if (type == INT)
            return intLabel_ < src.intLabel_;
        return symLabel_ < src.symLabel_;
    } else {
        return (type == INT);
    }
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
            intLabel_ = static_cast<int>(root_it->as_int());
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
