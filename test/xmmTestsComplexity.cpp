/*
 * xmmTestsComplexity.cpp
 *
 * Test suite for the algorithm complexity (Hierarchical HMM)
 *
 * Contact:
 * - Jules Francoise <jules.francoise@ircam.fr>
 *
 * This code has been initially authored by Jules Francoise
 * <http://julesfrancoise.com> during his PhD thesis, supervised by Frederic
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

#include "catch.hpp"
#include "xmmTestsUtilities.hpp"
#define XMM_TESTING
#include "xmm.h"
#include <iostream>
#include <random>

using namespace std;

// TEST_CASE("Profiling: Large number of classes", "[HierarhicalHMM]") {
//    xmm::TrainingSet ts;
//    int num_classes = 1200;
//    std::default_random_engine generator;
//    std::normal_distribution<float> dist;
//    for (int i = 0; i < num_classes; i++) {
//        ts.addPhrase(i, to_string(i + 1));
//        for (int frame_idx = 0; frame_idx < 200; frame_idx++) {
//            ts.getPhrase(i)->record({dist(generator)});
//        }
//    }
//    xmm::HierarchicalHMM model;
//    model.train(&ts);
//    model.reset();
//    for (int frame_idx = 0; frame_idx < 10000; frame_idx++) {
//        model.filter({dist(generator)});
//    }
//}
