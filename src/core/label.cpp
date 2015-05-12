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

xmm::Label::Label()
: type(INT),
  intLabel_(0),
  symLabel_("")
{}

xmm::Label::Label(int l)
: type(INT),
  intLabel_(l),
  symLabel_("")
{}

xmm::Label::Label(std::string l)
: type(SYM),
  intLabel_(0),
  symLabel_(l)
{}

xmm::Label::Label(char* l)
: type(SYM),
  intLabel_(0),
  symLabel_(l)
{}

xmm::Label::Label(Label const& src)
: type(src.type),
  intLabel_(src.intLabel_),
  symLabel_(src.symLabel_)
{}

xmm::Label& xmm::Label::operator=(Label const& src)
{
    if (this != &src) {
        this->type = src.type;
        this->intLabel_ = src.intLabel_;
        this->symLabel_ = src.symLabel_;
    }
    return *this;
}

xmm::Label& xmm::Label::operator=(int l)
{
    this->type = INT;
    this->intLabel_ = l;
    this->symLabel_ = "";
    return *this;
}

xmm::Label& xmm::Label::operator=(std::string l)
{
    this->type = SYM;
    this->intLabel_ = 0;
    this->symLabel_ = l;
    return *this;
}

xmm::Label& xmm::Label::operator=(char* l)
{
    this->type = SYM;
    this->intLabel_ = 0;
    this->symLabel_ = l;
    return *this;
}

bool xmm::Label::operator==(Label const& src) const
{
    if (!(this->type == src.type))
        return false;
    if (this->type == INT)
        return (this->intLabel_ == src.intLabel_);
    return (this->symLabel_ == src.symLabel_);
}

bool xmm::Label::operator!=(Label const& src) const
{
    return !operator==(src);
}

bool xmm::Label::operator<(Label const& src) const
{
    if (src.type == type) {
        if (type == INT)
            return intLabel_ < src.intLabel_;
        return symLabel_ < src.symLabel_;
    } else {
        return (type == INT);
    }
}

bool xmm::Label::operator<=(Label const& src) const
{
    return (operator<(src) || operator==(src));
}

bool xmm::Label::operator>(Label const& src) const
{
    return !operator<=(src);
}

bool xmm::Label::operator>=(Label const& src) const
{
    return !operator<(src);
}

int xmm::Label::getInt() const
{
    if (type != INT)
        throw std::runtime_error("Can't get INT from SYM label");
    return intLabel_;
}

std::string xmm::Label::getSym() const
{
    if (type != SYM)
        throw std::runtime_error("Can't get SYM from INT label");
    return symLabel_;
}

void xmm::Label::setInt(int l)
{
    type = INT;
    intLabel_ = l;
}

bool xmm::Label::trySetInt(std::string l)
{
    if (is_number(l)) {
        setInt(to_int(l));
        return true;
    }
    return false;
}


void xmm::Label::setSym(std::string l)
{
    type = SYM;
    symLabel_ = l;
}

void xmm::Label::setSym(char* l)
{
    type = SYM;
    symLabel_ = l;
}

JSONNode xmm::Label::to_json() const
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

void xmm::Label::from_json(JSONNode root)
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
    } catch (std::exception &e) {
        throw JSONException(e, root.name());
    }
}

std::string xmm::Label::as_string() const
{
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream& xmm::operator<<(std::ostream& stream, xmm::Label const& l)
{
    if (l.type == xmm::Label::INT)
        stream << l.getInt();
    else
        stream << l.getSym();
    return stream;
}

bool xmm::is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

int xmm::to_int(const std::string& s)
{
    std::istringstream myString(s);
    int value;
    myString >> value;
    return value;
}
