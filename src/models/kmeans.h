/*
 * kmeans.h
 *
 * K-Means clustering
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

#ifndef xmm_lib_kmeans__
#define xmm_lib_kmeans__

#include <stdio.h>
#include "probabilistic_model.h"

namespace xmm
{
    /**
     * @ingroup Core
     * @class KMeans
     * @brief K-Means Clustering algorithm
     */
    class KMeans : public Listener, public Writable {
    public:
        ///@cond DEVDOC
        
        static const unsigned int DEFAULT_MAX_ITERATIONS = 50;
        static const float DEFAULT_RELATIVE_VARIATION_THRESHOLD() { return 1e-20; }
        
        ///@endcond
        
        /**
         * @enum TRAINING_INIT
         * @brief Type of initizalization
         */
        typedef enum {
            /**
             * @brief random initialization (scaled using training set variance)
             */
            RANDOM,
            
            /**
             * @brief biased initialization: initialiazed with the first phrase
             */
            BIASED
        } TRAINING_INIT;
        
#pragma mark -
#pragma mark === Public Interface ===
#pragma mark > Constructors
        /** @name Constructors */
        ///@{
        
        /**
         * @brief Constructor
         * @param trainingSet training set associated with the model
         * @param nbClusters number of clusters
         */
        KMeans(TrainingSet *trainingSet = NULL,
               int nbClusters=1);
        /**
         * @brief Copy Constructor
         * @param src Source Model
         */
        KMeans(KMeans const& src);
        
        /**
         * @brief Assignment
         * @param src Source Model
         */
        KMeans& operator=(KMeans const& src);
        
        /**
         * @brief Destructor
         */
        virtual ~KMeans();
        ///@}
        
#pragma mark > Accessors
        /** @name Accessors */
        ///@{
        
        /**
         * @brief set the training set associated with the model
         * @details updates the dimensions of the model
         * @param trainingSet pointer to the training set.
         * @throws runtime_error if the training set has not the same number of modalities
         */
        void set_trainingSet(TrainingSet *trainingSet);
        
        /**
         * @brief Get Total Dimension of the model (sum of dimension of modalities)
         * @return total dimension of Gaussian Distributions
         */
        unsigned int dimension() const;
        
        /**
         * @brief Get Number of Clusters
         * @return Number of Clusters
         */
        unsigned int get_nbClusters();
        
        /**
         * @brief Set Number of Clusters
         * @param nbClusters Number of Clusters
         */
        void set_nbClusters(unsigned int nbClusters);
        
        /**
         * @brief Get Maximum number of training iterations
         * @return Maximum number of training iterations
         */
        unsigned int get_training_maxIterations() const;
        
        /**
         * @brief Set Maximum number of training iterations
         * @param maxIterations Maximum number of training iterations
         */
        void set_training_maxIterations(unsigned int maxIterations);
        
        /**
         * @brief Get relative distance Threshold for training
         * @return relative distance Threshold for training
         */
        unsigned int get_training_relativeDistanceThreshold() const;
        
        /**
         * @brief Set relative distance Threshold for training
         * @param threshold relative distance Threshold for training
         */
        void set_training_relativeDistanceThreshold(float threshold);
        ///@}
        
#pragma mark > Training
        /** @name Training */
        ///@{
        
        /**
         * @brief Main training method
         */
        void train();
        
        /**
         * @brief randomzie Cluster Centers (normalized width data variance)
         * of the first phrase of the training set
         */
        void randomizeClusters();
        
        /**
         * @brief set the callback function associated with the training algorithm
         * @details the function is called whenever the training is over or an error happened during training
         */
        void set_trainingCallback(void (*callback)(void *srcModel, ProbabilisticModel::CALLBACK_FLAG state, void* extradata), void* extradata);
        
        ///@}
        
#pragma mark > Performance
        /** @name Performance */
        ///@{
        
        /**
         * @brief Initialize the 'Performance' phase: prepare model for performance.
         */
        void performance_init();
        
        /**
         * @brief Main Performance function: updates the predictions of the model given a new observation
         * @param observation observation vector (must be of size 'dimension')
         */
        void performance_update(std::vector<float> const& observation);
        
        ///@}
        
#pragma mark > JSON I/O
        /** @name JSON I/O */
        ///@{
        
        /**
         * @brief Write to JSON Node
         * @return JSON Node containing training set information and data
         */
        JSONNode to_json() const;
        
        /**
         * @brief Read from JSON Node
         * @param root JSON Node containing training set information and data
         * @throws JSONException if the JSON Node has a wrong format
         */
        void from_json(JSONNode root);
        
        ///@}
        
#pragma mark -
#pragma mark === Public Attributes ===
        /**
         * Centers of the Clusters
         */
        std::vector<float> centers;
        
        /**
         * Number of training iterations
         */
        double trainingNbIterations;
        
        /**
         * Results: Distance of the observation to each cluster
         */
        std::vector<float> results_distances;
        
        /**
         * Results: Likeliest Cluster
         */
        unsigned int results_likeliest;
        
        /**
         * @brief Type of initialization for the K-Means Algorithm
         */
        KMeans::TRAINING_INIT trainingInitType;
        
    protected:
        ///@cond DEVDOC
        
#pragma mark -
#pragma mark === Protected Methods ===
        /**
         * @brief handle notifications of the training set
         * @details here only the dimensions attributes of the training set are considered
         * @param attribute name of the attribute: should be either "dimension" or "dimension_input"
         */
        void notify(std::string attribute);
        
#pragma mark > training
        /** @name Training (protected) */
        ///@{
        
        /**
         * @brief Initialize the clusters using a regular segmentation
         * of the first phrase of the training set
         */
        void initClustersWithFirstPhrase();
        
        /**
         * @brief Update method for training
         * @details computes the cluster associated with each data points, and update
         * Cluster centers
         */
        void updateCenters(std::vector<float>& previous_centers);
        
        ///@}
        
#pragma mark -
#pragma mark === Protected Attributes ===
        /**
         * Pointer to the training set.
         */
        TrainingSet *trainingSet;
        
        /**
         * Number of Clusters
         */
        unsigned int nbClusters_;
        
        /**
         * Dimension of the data
         */
        unsigned int dimension_;
        
        /**
         * Maximum number of training iterations
         */
        unsigned int training_maxIterations_;
        
        /**
         * Threshold for training: stop training when the maximum relative
         * center variation (maximum variation of centers' position divided
         * by mean distance between clusters) gets under this threshold.
         */
        float training_relativeDistanceThreshold_;
        
        /**
         * @brief Callback function for the training algorithm
         */
        void (*trainingCallbackFunction_)(void *srcModel, ProbabilisticModel::CALLBACK_FLAG state, void* extradata);
        
        /**
         * @brief Extra data to pass in argument to the callback function
         */
        void *trainingExtradata_;
        
        ///@endcond
    };
    
    ///@cond DEVDOC
    
    /**
     * @brief Simple Euclidian distance measure
     * @param vector1 first data point
     * @param vector2 first data point
     * @param dimension dimension of the data space
     * @return euclidian distance between the 2 points
     */
    template <typename T>
    T euclidian_distance(const T* vector1,
                         const T* vector2,
                         unsigned int dimension);
    
    ///@endcond
    
}

#endif
