/*
 * xmmTestsUtilities.hpp
 *
 * Utilities for Unit-Testing
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

#ifndef xmm_lib_catch_utilities_h
#define xmm_lib_catch_utilities_h

#include "catch.hpp"

/**
 * @brief Check for vector approximate equality
 */
template <typename T>
void CHECK_VECTOR_APPROX(std::vector<T> const& a, std::vector<T> const& b, double epsilon=-1.) {
    REQUIRE(a.size() == b.size());
    std::string errormsg = "CHECK_VECTOR_APPROX:\n{";
    std::string errormsg2 = "}\n==\n{";
    std::ostringstream convert;
    std::ostringstream convert2;
    for (size_t i=0; i<a.size(); ++i) {
        convert << a[i] << " ";
        convert2 << b[i] << " ";
    }
    INFO(errormsg + convert.str() + errormsg2 + convert2.str() );
    if (epsilon > 0.0) {
        for (size_t i=0; i<a.size(); ++i) {
            REQUIRE(a[i] == Approx(b[i]).epsilon(epsilon));
        }
    } else {
        for (size_t i=0; i<a.size(); ++i) {
            REQUIRE(a[i] == Approx(b[i]));
        }
    }
};

#endif
