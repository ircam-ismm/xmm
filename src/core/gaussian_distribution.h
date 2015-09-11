/*
 * gaussian_distribution.h
 *
 * Multivariate Gaussian Distribution
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

#ifndef xmm_lib_gaussian_distribution__
#define xmm_lib_gaussian_distribution__

#include "json_utilities.h"
#include "xmm_common.h"

namespace xmm
{
    /**
     * @ingroup Utilities
     * @struct Ellipse
     * @brief Simple structure for storing Ellipse parameters
     */
    struct Ellipse {
        /**
         * @brief x center position
         */
        float x;
        
        /**
         * @brief y center position
         */
        float y;
        
        /**
         * @brief width: minor axis length
         */
        float width;
        
        /**
         * @brief height: major axis length
         */
        float height;
        
        /**
         * @brief angle (radians)
         */
        float angle;
    };
    
    /**
     * @ingroup Core
     * @class GaussianDistribution
     * @brief Multivariate Gaussian Distribution
     * @details Full covariance, optionally multimodal with support for regression
     */
    class GaussianDistribution : public Writable
    {
    public:
        ///@cond DEVDOC
        
        /**
         * default offset for covariance matrix
         */
        ///@{
        static const double DEFAULT_VARIANCE_OFFSET_RELATIVE() { return 1.e-2; }
        static const double DEFAULT_VARIANCE_OFFSET_ABSOLUTE() { return 1.e-3; }
        ///@}
        
        ///@endcond
        
        /**
         * @brief Covariance Mode
         */
        enum COVARIANCE_MODE {
            /**
             * @brief Full covariance
             */
            FULL = 0,
            
            /**
             * @brief Diagonal covariance (diagonal matrix)
             */
            DIAGONAL = 1
        };
        
#pragma mark > Constructors
        /** @name Constructors */
        ///@{
        
        /**
         * @brief Default Constructor
         * @param flags construction flags. Use the flag 'BIMODAL' for use with regression.
         * @param dimension dimension of the distribution
         * @param offset_relative Offset added to diagonal covariance (proportional to variance)
         * @param offset_absolute Offset added to diagonal covariance (minimum value)
         * @param dimension_input dimension of the input modality in bimodal mode.
         * @param covariance_mode covariance mode (full vs diagonal)
         */
        GaussianDistribution(xmm_flags flags = NONE,
                             unsigned int dimension=1,
                             unsigned int dimension_input = 0,
                             double offset_relative = DEFAULT_VARIANCE_OFFSET_RELATIVE(),
                             double offset_absolute = DEFAULT_VARIANCE_OFFSET_ABSOLUTE(),
                             COVARIANCE_MODE covariance_mode = FULL);
        
        /**
         * @brief Copy constructor
         * @param src source distribution
         */
        GaussianDistribution(GaussianDistribution const& src);
        
        /**
         * @brief Assignment
         * @param src source distribution
         */
        GaussianDistribution& operator=(GaussianDistribution const& src);
        
        /**
         * @brief Destructor
         */
        virtual ~GaussianDistribution();
        
        ///@}
        
#pragma mark > Accessors
        /** @name Accessors */
        ///@{
        
        /**
         * @brief Get Dimension of the distribution
         * @return dimension
         */
        unsigned int dimension() const;
        
        /**
         * @brief Set Dimension of the distribution
         */
        void set_dimension(unsigned int dimension);
        
        /**
         * @brief Get Dimension of the input modality
         * @return dimension
         */
        unsigned int dimension_input() const;
        
        /**
         * @brief Set Dimension of the input modality
         * @param dimension_input dimension of the input modality
         * @throws out_of_range if the dimension is superior to the total dimension
         */
        void set_dimension_input(unsigned int dimension_input);
        
        /**
         * @brief get the current covariance mode
         */
        COVARIANCE_MODE get_covariance_mode() const;
        
        /**
         * @brief set the covariance mode
         * @param covariance_mode target covariance mode
         */
        void set_covariance_mode(COVARIANCE_MODE covariance_mode);
        
        ///@}
        
#pragma mark > Likelihood & Regression
        /** @name Likelihood & Regression */
        ///@{
        
        /**
         * @brief Get Likelihood of a data vector
         * @param observation data observation (must be of size @a dimension)
         * @return likelihood
         * @throws runtime_error if the Covariance Matrix is not invertible
         */
        double likelihood(const float* observation) const;
        
        /**
         * @brief Get Likelihood of a data vector for input modality
         * @param observation_input observation (must be of size @a dimension_input)
         * @return likelihood
         * @throws runtime_error if the Covariance Matrix of the input modality is not invertible
         * @throws runtime_error if the model is not bimodal
         */
        double likelihood_input(const float* observation_input) const;
        
        /**
         * @brief Get Likelihood of a data vector for bimodal mode
         * @param observation_input observation of the input modality
         * @param observation_output observation of the output modality
         * @throws runtime_error if the Covariance Matrix is not invertible
         * @throws runtime_error if the model is not bimodal
         * @return likelihood
         */
        double likelihood_bimodal(const float* observation_input, const float* observation_output) const;
        
        /**
         * @brief Linear Regression using the Gaussian Distribution (covariance-based)
         * @param observation_input input observation (must be of size: @a dimension_input)
         * @param predicted_output predicted output vector (size: dimension-dimension_input)
         * @throws runtime_error if the model is not bimodal
         */
        void regression(std::vector<float> const& observation_input, std::vector<float>& predicted_output) const;
        
        ///@}
        
#pragma mark > JSON I/O
        /** @name JSON I/O */
        ///@{
        
        /**
         * @brief Write to JSON Node
         * @return The JSON Node containing the Gaussian Distribution parameters
         */
        JSONNode to_json() const;
        
        /**
         * @brief Write to JSON Node
         * @details allocate model parameters and updates inverse Covariances
         * @param root JSON Node containing model information and parameters
         * @throws JSONException if the JSONNode has a wrong format
         */
        void from_json(JSONNode root);
        
        ///@}
        
#pragma mark > Utilities
        /** @name Utilities */
        ///@{
        
        /**
         * @brief Add @a offset to the diagonal of the covariance matrix
         * @details Ensures convergence + generalization on few examples
         */
        void addOffset();
        
        /**
         * @brief Compute inverse covariance matrix
         * @throws runtime_error if the covariance matrix is not invertible
         */
        void updateInverseCovariance();
        
        /**
         * @brief Compute the conditional variance vector of the output modality
         * (conditioned over the input).
         * @throws runtime_error if the model is not bimodal
         */
        void updateOutputVariances();
        
        /**
         * @brief Compute the 95% Confidence Interval ellipse of the Gaussian
         * @details the ellipse is 2D, and is therefore projected over 2 axes
         * @param dimension1 index of the first axis
         * @param dimension2 index of the second axis
         * @throws out_of_range if the dimensions are out of bounds
         * @return ellipse parameters
         */
        Ellipse ellipse(unsigned int dimension1,
                        unsigned int dimension2);
        
        ///@}
        
#pragma mark > Conversion & Extraction
        /** @name Conversion & Extraction */
        ///@{
        
        /**
         * @brief Convert to bimodal distribution in place
         * @param dimension_input dimension of the input modality
         * @throws runtime_error if the model is already bimodal
         * @throws out_of_range if the requested input dimension is too large
         */
        void make_bimodal(unsigned int dimension_input);
        
        /**
         * @brief Convert to unimodal distribution in place
         * @throws runtime_error if the model is already unimodal
         */
        void make_unimodal();
        
        /**
         * @brief extract a sub-distribution with the given columns
         * @param columns columns indices in the target order
         * @throws runtime_error if the model is training
         * @throws out_of_range if the number or indices of the requested columns exceeds the current dimension
         * @return a Gaussian Distribution from the current model considering only the target columns
         */
        GaussianDistribution extract_submodel(std::vector<unsigned int>& columns) const;
        
        /**
         * @brief extract the sub-distribution of the input modality
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a unimodal Gaussian Distribution of the input modality from the current bimodal model
         */
        GaussianDistribution extract_submodel_input() const;
        
        /**
         * @brief extract the sub-distribution of the output modality
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a unimodal Gaussian Distribution of the output modality from the current bimodal model
         */
        GaussianDistribution extract_submodel_output() const;
        
        /**
         * @brief extract the model with reversed input and output modalities
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a bimodal Gaussian Distribution  that swaps the input and output modalities
         */
        GaussianDistribution extract_inverse_model() const;
        
        ///@}
        
#pragma mark -
#pragma mark === Public Attributes ===
        /**
         * @brief Mean of the Gaussian Distribution
         */
        std::vector<double> mean;
        
        /**
         * @brief Covariance Matrix of the Gaussian Distribution
         */
        std::vector<double> covariance;
        
        /**
         * @brief Offset added to diagonal covariance (proportional to variance)
         */
        double offset_relative;
        
        /**
         * @brief Offset added to diagonal covariance (minimum value)
         */
        double offset_absolute;
        
        /**
         * @brief Scaling of each dimension of the Gaussian Distribution (used of variance offsets)
         */
        std::vector<float> scale;
        
        /**
         * @brief Conditional Output Variance
         * updated when covariances matrices are inverted.
         */
        std::vector<double> output_variance;
        
#ifndef XMM_TESTING
    private:
#endif
#pragma mark -
#pragma mark === Private Methods ===
#pragma mark > Utilities
        /**
         * @brief Copy between 2 Gaussian Distributions
         * @param dst destination distribution
         * @param src source distribution
         */
        void _copy(GaussianDistribution *dst, GaussianDistribution const& src);
        
        /**
         * @brief Resize Mean and Covariance Vectors to appropriate dimension.
         */
        void allocate();
        
#pragma mark -
#pragma mark === Private Attributes ===
        /**
         * @brief Defines if regression parameters need to be computed
         */
        bool bimodal_;
        
        /**
         * @brief Dimension of the distribution
         */
        unsigned int dimension_;
        
        /**
         * @brief Dimension of the input modality
         */
        unsigned int dimension_input_;
        
        /**
         * @brief Determinant of the covariance matrix
         */
        double covarianceDeterminant;
        
        /**
         * @brief Inverse covariance matrix
         */
        std::vector<double> inverseCovariance_;
        
        /**
         * @brief Determinant of the covariance matrix of the input modality
         */
        double covarianceDeterminant_input_;
        
        /**
         * @brief Inverse covariance matrix of the input modality
         */
        std::vector<double> inverseCovariance_input_;
        
        /**
         * @brief Covariance Mode
         */
        COVARIANCE_MODE covariance_mode_;
    };
    
}

#endif
