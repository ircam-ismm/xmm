/*
 * xmm.h
 *
 * XMM - Probabilistic Models for Continuous Motion Recognition and Mapping
 * ============================================================================
 *
 * XMM is a portable, cross-platform C++ library that implements Gaussian
 * Mixture Models and Hidden Markov Models for recognition and regression.
 * The XMM library was developed for movement interaction in creative
 * applications and implements an interactive machine learning workflow with
 * fast training and continuous, real-time inference.
 *
 * Contact:
 * - Jules Françoise: <jules.francoise@ircam.fr>
 *
 *
 * Authors:
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
 * This project is released under the GPLv3 license.
 * For commercial applications, a proprietary license is available upon
 * request to Frederick Rousseau <frederick.rousseau@ircam.fr>.
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
 *
 * Citing this work:
 * If you use this code for research purposes, please cite one of the following publications:
 *
 * - J. Françoise, N. Schnell, R. Borghesi, and F. Bevilacqua,
 *   Probabilistic Models for Designing Motion and Sound Relationships.
 *   In Proceedings of the 2014 International Conference on New Interfaces
 *   for Musical Expression, NIME’14, London, UK, 2014.
 * - J. Françoise, N. Schnell, and F. Bevilacqua, A Multimodal Probabilistic
 *   Model for Gesture-based Control of Sound Synthesis. In Proceedings of the
 *   21st ACM international conference on Multimedia (MM’13), Barcelona,
 *   Spain, 2013.
 */

#ifndef xmm_lib_xmm_h
#define xmm_lib_xmm_h

#include "models/kmeans.h"
#include "models/gmm.h"
#include "models/gmm_group.h"
#include "models/hmm.h"
#include "models/hierarchical_hmm.h"

#endif
