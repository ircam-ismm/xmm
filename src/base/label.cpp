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