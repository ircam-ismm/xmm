//
//  catch_utilities.h
//  xmm-lib
//
//  Created by Jules Francoise on 06/04/2015.
//
//

#ifndef xmm_lib_catch_utilities_h
#define xmm_lib_catch_utilities_h

#include "catch.hpp"
using namespace std;

/**
 * @brief Check for vector approximate equality
 */
template <typename T>
void CHECK_VECTOR_APPROX(vector<T> const& a, vector<T> const& b, double epsilon=-1.) {
    REQUIRE(a.size() == b.size());
    string errormsg = "CHECK_VECTOR_APPROX:\n{";
    string errormsg2 = "}\n==\n{";
    ostringstream convert;
    ostringstream convert2;
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
