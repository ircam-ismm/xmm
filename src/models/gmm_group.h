/*
 * gmm_group.h
 *
 * Group of Gaussian Mixture Models for continuous recognition and regression with multiple classes
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

#ifndef xmm_lib_gmm_group_h
#define xmm_lib_gmm_group_h

#include "model_group.h"
#include "gmm.h"

namespace xmm
{
    
    /**
     * @ingroup GMM
     * @class GMMGroup
     * @brief Set of GMMs Running in parallel
     * @details Allows to perform GMM-based pattern recognition.
     * @see ModelGroup
     */
    class GMMGroup : public ModelGroup< GMM > {
    public:
        /**
         * @brief Iterator over models
         */
        typedef std::map<Label, GMM>::iterator model_iterator;
        
        /**
         * @brief Constant Iterator over models
         */
        typedef std::map<Label, GMM>::const_iterator const_model_iterator;
        
#pragma mark > Constructors
        /*@{*/
        /** @name Constructors */
        /**
         * @brief Constructor
         * @param globalTrainingSet training set associated with the model
         * @param flags Construction Flags: use 'BIMODAL' for use with Gaussian Mixture Regression.
         * @param covariance_mode covariance mode (full vs diagonal)
         */
        GMMGroup(xmm_flags flags = NONE,
                 TrainingSet *globalTrainingSet=NULL,
                 GaussianDistribution::COVARIANCE_MODE covariance_mode = GaussianDistribution::FULL);
        
        /**
         * @brief Copy Constructor
         * @param src Source Model
         */
        GMMGroup(GMMGroup const& src);
        
        /**
         * @brief Assignment
         * @param src Source Model
         */
        GMMGroup& operator=(GMMGroup const& src);
        
        /*@}*/
        
#pragma mark > Accessors
        /*@{*/
        /** @name Accessors */
        /**
         * @brief Get the number of Gaussian mixture Components
         * @return number of Gaussian mixture components
         */
        int get_nbMixtureComponents() const;
        
        /**
         * @brief Get Offset added to covariance matrices for convergence (Relative to data variance)
         * @return Offset added to covariance matrices for convergence
         */
        double get_varianceOffset_relative() const;
        
        /**
         * @brief Get Offset added to covariance matrices for convergence (Minimum value)
         * @return Offset added to covariance matrices for convergence
         */
        double get_varianceOffset_absolute() const;
        
        /**
         * @brief Set the number of mixture components of the model
         * @warning sets the model to be untrained.
         * @param nbMixtureComponents number of Gaussian Mixture Components
         * @throws invalid_argument if nbMixtureComponents is <= 0
         */
        void set_nbMixtureComponents(int nbMixtureComponents);
        
        /**
         * @brief Set the offset to add to the covariance matrices
         * @param varianceOffset_relative offset to add to the diagonal of covariance matrices (relative to data variance)
         * @param varianceOffset_absolute offset to add to the diagonal of covariance matrices (minimum value)
         * @throws invalid_argument if the covariance offset is <= 0
         */
        void set_varianceOffset(double varianceOffset_relative, double varianceOffset_absolute);
        
        /**
         * @brief get the current covariance mode
         */
        GaussianDistribution::COVARIANCE_MODE get_covariance_mode() const;
        
        /**
         * @brief set the covariance mode
         * @param covariance_mode target covariance mode
         */
        void set_covariance_mode(GaussianDistribution::COVARIANCE_MODE covariance_mode);
        
        /*@}*/
        
#pragma mark > Performance
        /*@{*/
        /** @name Performance */
        /**
         * @brief Main Play function: performs recognition (unimodal mode) and regression (bimodal mode)
         * @details The predicted output is stored in the observation vector in bimodal mode
         * @param observation observation vector
         */
        void performance_update(std::vector<float> const& observation);
        
        /*@}*/
        
#pragma mark > JSON I/O
        /*@{*/
        /** @name JSON I/O */
        /**
         * @brief Write to JSON Node
         * @return JSON Node containing training set information and data
         */
        virtual JSONNode to_json() const;
        
        /**
         * @brief Read from JSON Node
         * @param root JSON Node containing training set information and data
         * @throws JSONException if the JSON Node has a wrong format
         */
        virtual void from_json(JSONNode root);
        
        /*@}*/
        
#pragma mark > Conversion & Extraction
        /*@{*/
        /** @name Conversion & Extraction */
        
        /**
         * @brief Convert to bimodal GMMGroup in place
         * @param dimension_input dimension of the input modality
         * @throws runtime_error if the model is already bimodal
         * @throws out_of_range if the requested input dimension is too large
         */
        void make_bimodal(unsigned int dimension_input);
        
        /**
         * @brief Convert to unimodal GMMGroup in place
         * @throws runtime_error if the model is already unimodal
         */
        void make_unimodal();
        
        /**
         * @brief extract a submodel with the given columns
         * @param columns columns indices in the target order
         * @throws runtime_error if the model is training
         * @throws out_of_range if the number or indices of the requested columns exceeds the current dimension
         * @return a GMMGroup from the current model considering only the target columns
         */
        GMMGroup extract_submodel(std::vector<unsigned int>& columns) const;
        
        /**
         * @brief extract the submodel of the input modality
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a unimodal GMMGroup of the input modality from the current bimodal model
         */
        GMMGroup extract_submodel_input() const;
        
        /**
         * @brief extract the submodel of the output modality
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a unimodal GMMGroup of the output modality from the current bimodal model
         */
        GMMGroup extract_submodel_output() const;
        
        /**
         * @brief extract the model with reversed input and output modalities
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a bimodal GMMGroup  that swaps the input and output modalities
         */
        GMMGroup extract_inverse_model() const;
        
        /*@}*/
        
    protected:
        /**
         * @brief Copy between two Hierarhical HMMs
         * @param src Source Model
         * @param dst Destination Model
         */
        virtual void _copy(GMMGroup *dst, GMMGroup const& src);
    };
    
}

#endif
