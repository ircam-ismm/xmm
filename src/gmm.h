//
//  gmm.h
//  mhmm
//
//  Created by Jules Francoise on 08/08/13.
//
//

#ifndef mhmm_gmm_h
#define mhmm_gmm_h

#include "em_based_learning_model.h"
#include "matrix.h"

#define GMM_DEFAULT_NB_MIXTURE_COMPONENTS 1
#define GMM_DEFAULT_COVARIANCE_OFFSET 0.01

/*!
 * @class GMM
 * @brief Gaussian Mixture Model
 *
 * Can be either autonomous or a state of a HMM: defines observation probabilities for each state
 @todo detail documentation
 @tparam ownData defines if phrases has own data or shared memory
 */
template<bool ownData>
class GMM : public EMBasedLearningModel< Phrase<ownData, 1>, int> {
public:
    typedef typename map<int, Phrase<ownData, 1>* >::iterator phrase_iterator;
    
    vector<float> mean;
    vector<float> covariance;
    vector<float> mixtureCoeffs;
    
#pragma mark -
#pragma mark Constructors
    /*! @name Constructors */
    /*!
     Constructor
     @param \_trainingSet training set associated with the model
     @param nbMixtureComponents\_ number of mixture components
     @param covarianceOffset_ offset to add to the diagonal of covariances matrices (useful to guarantee convergence)
     */
    GMM(TrainingSet<Phrase<ownData, 1>, int> *_trainingSet=NULL,
                  int nbMixtureComponents_ = GMM_DEFAULT_NB_MIXTURE_COMPONENTS,
                  float covarianceOffset_= GMM_DEFAULT_COVARIANCE_OFFSET)
    : EMBasedLearningModel<Phrase<ownData, 1>, int>(_trainingSet)
    {
        nbMixtureComponents  = nbMixtureComponents_;
        covarianceOffset     = covarianceOffset_;
        
        if (this->trainingSet) {
            dimension = this->trainingSet->get_dimension();
        } else {
            dimension = 1;
        }
        
        reallocParameters();
        initTraining();
    }
    
    /*!
     Copy constructor
     */
    GMM(GMM const& src) : EMBasedLearningModel< Phrase<ownData, 1>, int>(src)
    {
        copy(this, src);
    }
    
    /*!
     Assignment
     */
    GMM& operator=(GMM const& src)
    {
        if(this != &src)
        {
            copy(this, src);
        }
        return *this;
    };
    
    /*!
     Copy between 2 MultimodalGMM models
     */
    void copy(GMM *dst, GMM const& src)
    {
        EMBasedLearningModel<Phrase<ownData, 1>, int>::copy(dst, src);
        dst->nbMixtureComponents     = src.nbMixtureComponents;
        dst->covarianceOffset        = src.covarianceOffset;
        dst->covarianceDeterminant   = src.covarianceDeterminant;
        
        dst->dimension = src.dimension;
        
        dst->mixtureCoeffs = src.mixtureCoeffs;
        dst->mean = src.mean;
        dst->covariance = src.covariance;
        dst->inverseCovariance = src.inverseCovariance;
        dst->covarianceDeterminant = src.covarianceDeterminant;
        
        dst->reallocParameters();
    }
    
    /*!
     Destructor
     */
    ~GMM()
    {
        mean.clear();
        covariance.clear();
        inverseCovariance.clear();
        covarianceDeterminant.clear();
        mixtureCoeffs.clear();
    }
    
    /*!
     handle notifications of the training set
     
     here only the dimensions attributes of the training set are considered
     */
    void notify(string attribute)
    {
        if (!this->trainingSet) return;
        if (attribute == "dimension") {
            dimension = this->trainingSet->get_dimension();
            reallocParameters();
            return;
        }
    }
    
    /*!
     Set training set associated with the model
     */
    void set_trainingSet(TrainingSet<Phrase<ownData, 1>, int> *_trainingSet)
    {
        this->trainingSet = _trainingSet;
        if (this->trainingSet) {
            dimension = this->trainingSet->get_dimension();
        } else {
            dimension = 1;
        }
        reallocParameters();
    }
    
#pragma mark -
#pragma mark Parameters
    /*! @name Parameters */
    /*!
     Re-allocate model parameters
     */
    void reallocParameters()
    {
        mean.resize(nbMixtureComponents*dimension);
        covariance.resize(nbMixtureComponents*dimension*dimension);
        inverseCovariance.resize(nbMixtureComponents*dimension*dimension);
        mixtureCoeffs.resize(nbMixtureComponents);
        covarianceDeterminant.resize(nbMixtureComponents);
        beta.resize(nbMixtureComponents);
    }
    
    /*!
     Initialize the means of the gaussian components with the first phrase of the training set
     */
    void initMeansWithFirstPhrase()
    {
        if (!this->trainingSet) return;
        int nbPhrases = this->trainingSet->size();
        
        if (nbPhrases == 0) return;
        int step = this->trainingSet->begin()->second->getlength() / nbMixtureComponents;
		
        int offset(0);
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d=0; d<dimension; d++) {
                meanOfComponent(c)[d] = 0.0;
            }
            for (int t=0; t<step; t++) {
                for (int d=0; d<dimension; d++) {
                    meanOfComponent(c)[d] += (*this->trainingSet->begin()->second)(offset+t, d) / float(step);
                }
            }
            offset += step;
        }
    }
    
    /*!
     Set model parameters to zero, except means (sometimes means are not re-estimated in the training algorithm)
     */
    void setParametersToZero()
    {
        for (int c=0; c<nbMixtureComponents; c++) {
            mixtureCoeffs[c] = 0.;
            for (int d=0; d<dimension; d++) {
                for (int d2=0; d2<dimension; d2++) {
                    covariance[c*dimension*dimension+d*dimension+d2] = 0.;
                }
            }
        }
    }
    
    /*!
     Normalize mixture coefficients
     */
    void normalizeMixtureCoeffs()
    {
        double norm_const(0.);
        for (int c=0; c<nbMixtureComponents; c++) {
            norm_const += mixtureCoeffs[c];
        }
        if (norm_const > 0) {
            for (int c=0; c<nbMixtureComponents; c++) {
                mixtureCoeffs[c] /= norm_const;
            }
        } else {
            for (int c=0; c<nbMixtureComponents; c++) {
                mixtureCoeffs[c] = 1/float(nbMixtureComponents);
            }
        }
    }
    
    /*!
     add offset to the diagonal of the covariance matrices => guarantee convergence through covariance matrix invertibility
     */
    void addCovarianceOffset()
    {
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d=0; d<dimension; d++)
                covarianceOfComponent(c)[d*dimension+d] += covarianceOffset;
        }
    }
    
#pragma mark -
#pragma mark Accessors
    /*! @name Accessors */
    int get_dimension() const
    {
        return dimension;
    }
    
    int   get_nbMixtureComponents() const
    {
        return nbMixtureComponents;
    }
    
    float get_covarianceOffset() const
    {
        return covarianceOffset;
    }
    
    /*!
     Set the number of mixture components of the model
     */
    void set_nbMixtureComponents(int nbMixtureComponents_)
    {
        if (nbMixtureComponents_ < 1) throw RTMLException("Number of mixture components must be > 0", __FILE__, __FUNCTION__, __LINE__);
        if (nbMixtureComponents_ == nbMixtureComponents) return;
        
        nbMixtureComponents = nbMixtureComponents_;
        reallocParameters();
        this->trained = false;
    }
    
    /*!
     Set the offset to add to the covariance matrices
     */
    void set_covarianceOffset(float covarianceOffset_)
    {
        if (covarianceOffset_ <= 0.) throw RTMLException("Covariance offset must be > 0", __FILE__, __FUNCTION__, __LINE__);
        
        covarianceOffset = covarianceOffset_;
    }
    
    // Observation probabilities
    // ================================
#pragma mark -
#pragma mark Observation probabilities
    /*! @name Observation probabilities */
    /*!
     Observation probability
     @param obs observation vector
     @param mixtureComponent index of the mixture component. if unspecified or negative,
     full mixture observation probability is computed
     */
    double obsProb(const float *obs, int mixtureComponent=-1)
    {
        double p(0.);
        
        if (mixtureComponent < 0) {
            for (mixtureComponent=0; mixtureComponent<nbMixtureComponents; mixtureComponent++) {
                p += obsProb(obs, mixtureComponent);
            }
        } else {
            
            p = mixtureCoeffs[mixtureComponent] * gaussianProbabilityFullCovariance(obs,
                                                                                    meanOfComponent(mixtureComponent),
                                                                                    covarianceDeterminant[mixtureComponent],
                                                                                    inverseCovarianceOfComponent(mixtureComponent),
                                                                                    dimension);
        }
        
        return p;
    }
        
    // Utility
    // ================================
#pragma mark -
#pragma mark Utility
    /*! @name Utility */
    /*!
     get mean of a particular gaussian component (multimodal)
     */
    vector<float>::iterator meanOfComponent(int component)
    {
        return mean.begin() + component * dimension;
    }
    
    /*!
     get covariance of a particular gaussian component (multimodal)
     */
    vector<float>::iterator covarianceOfComponent(int component)
    {
        return covariance.begin() + component * dimension * dimension;
    }
    
    /*!
     get inverse covariance of a particular gaussian component (multimodal)
     */
    vector<float>::iterator inverseCovarianceOfComponent(int component)
    {
        return inverseCovariance.begin() + component * dimension * dimension;
    }
        
    /*!
     get index of the likeliest component
     */
    int likeliestComponent()
    {
        int component(0);
        double maxProb = mixtureCoeffs[component] / sqrt(covarianceDeterminant[component]);
        double prob;
        for (int c=1 ; c<nbMixtureComponents; c++) {
            prob = mixtureCoeffs[c] / sqrt(covarianceDeterminant[c]);
            if (prob > maxProb) {
                component = c;
                maxProb = prob;
            }
        }
        return component;
    }
    
    /*!
     Compute likeliest component given an observation. The likelihood is computed using observation probabilities
     and mixture coefficients
     */
    int likeliestComponent(const float *obs)
    {
        // !! here, the likeliest component is computed with the mixture coeffs => relevant ?
        int component(0);
        double maxProb = obsProb(obs, component);
        double prob;
        for (int c=1 ; c<nbMixtureComponents; c++) {
            prob = obsProb(obs, c);
            if (prob > maxProb) {
                component = c;
                maxProb = prob;
            }
        }
        return component;
    }
    
    /*!
     update inverse covariances of each gaussian component
     */
    void updateInverseCovariances()
    {
        Matrix<float> cov_matrix(dimension, dimension, false);
        
        Matrix<float> *inverseMat;
        double det;
        
        for (int c=0; c<nbMixtureComponents; c++)
        {
            // Update inverse covariance for gesture+sound
            cov_matrix.data = covarianceOfComponent(c);
            inverseMat = cov_matrix.pinv(&det);
            covarianceDeterminant[c] = det;
            vectorCopy(inverseCovarianceOfComponent(c), inverseMat->data, dimension*dimension);
            delete inverseMat;
        }
    }
    
#pragma mark -
#pragma mark Gaussian Mixture Regression
    /*! @name Gaussian Mixture Regression */
    /*!
     Compute gaussian mixture regression
     @param obs multimodal gesture-sound observation vector, gesture part is used for prediction.
     the estimated sound vector is stored in the sound part.
     @return likelihood computed over the gesture part
     */
    double recognition(float *obs)
    {
        recognition_beta(obs);
        
        return obsProb(obs, -1);
    }
        
    /*!
     Compute likelihoods of each components given a gesture observation vector
     */
    void recognition_beta(const float *obs)
    {
        double norm_const(0.);
        for (int c=0; c<nbMixtureComponents; c++) {
            beta[c] = obsProb(obs, c);
            norm_const += beta[c];
        }
        for (int c=0; c<nbMixtureComponents; c++) {
            beta[c] /= norm_const;
        }
    }
        
#pragma mark -
#pragma mark Play!
    /*! @name Playing */
    /*!
     initialize playing mode
     */
    void initPlaying()
    {}
    
    /*!
     play function: estimate sound parameters using gesture input and compute likelihood
     */
    double play(float *obs)
    {
        double prob = recognition(obs);
        return this->updateLikelihoodBuffer(prob);
    }
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    void write(ostream& outStream, bool writeTrainingSet=false)
    {
        outStream << "# Multimodal GMM \n";
        outStream << "# =========================================\n";
        EMBasedLearningModel<Phrase<ownData, 1>, int>::write(outStream, writeTrainingSet);
        outStream << "# Dimension\n";
        outStream << dimension << endl;
        outStream << "# Number of mixture Components\n";
        outStream << nbMixtureComponents << endl;
        outStream << "# Covariance Offset\n";
        outStream << covarianceOffset << endl;
        outStream << "# Mixture Coefficients\n";
        for (int c=0 ; c<nbMixtureComponents; c++) {
            outStream << mixtureCoeffs[c] << " ";
        }
        outStream << endl;
        outStream << "# Mean\n";
        for (int c=0 ; c<nbMixtureComponents; c++) {
            for (int d=0; d<dimension; d++) {
                outStream << meanOfComponent(c)[d] << " ";
            }
            outStream << endl;
        }
        outStream << "# Covariance\n";
        for (int c=0 ; c<nbMixtureComponents; c++) {
            for (int d1=0; d1<dimension; d1++) {
                for (int d2=0; d2<dimension; d2++) {
                    outStream << covarianceOfComponent(c)[d1*dimension+d2] << " ";
                }
                outStream << endl;
            }
        }
    }
    
    void read(istream& inStream, bool readTrainingSet=false)
    {
        EMBasedLearningModel<Phrase<ownData, 1>, int>::read(inStream, readTrainingSet);
        
        // Get Dimensions
        skipComments(&inStream);
        inStream >> dimension;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);;
        
        // Get Number of mixture components
        skipComments(&inStream);
        inStream >> nbMixtureComponents;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);;
        
        reallocParameters();
        
        // Get Covariance Offset
        skipComments(&inStream);
        inStream >> covarianceOffset;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);;
        
        // Get Mixture coefficients
        skipComments(&inStream);
        for (int c=0 ; c<nbMixtureComponents; c++) {
            inStream >> mixtureCoeffs[c];
            if (!inStream.good())
                throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);;
        }
        
        // Get Mean
        skipComments(&inStream);
        for (int c=0 ; c<nbMixtureComponents; c++) {
            for (int d=0; d<dimension; d++) {
                inStream >> meanOfComponent(c)[d];
                if (!inStream.good())
                    throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);;
            }
        }
        
        // Get Covariance
        skipComments(&inStream);
        for (int c=0 ; c<nbMixtureComponents; c++) {
            for (int d1=0; d1<dimension; d1++) {
                for (int d2=0; d2<dimension; d2++) {
                    inStream >> covarianceOfComponent(c)[d1*dimension+d2];
                    if (!inStream.good())
                        throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);;
                }
            }
        }
        
        updateInverseCovariances();
    }
    
#pragma mark -
#pragma mark Debug
    /*! @name Debug */
    void dump()
    {
        if (this->trainingSet) {
            int nbPhrases = this->trainingSet->size();
            cout << "Number of phrases = " << nbPhrases << endl;
            for (phrase_iterator it = this->trainingSet->begin() ; it != this->trainingSet->end() ; it++) {
                cout << "size of phrase " << it->first << " = " << it->second->getlength() << endl;
                // cout << "phrase " << it->first << ": data = \n";
                // it->second->print();
            }
            cout << "\n\n";
        }
        
        cout << "Dimension = " << dimension << endl;
        cout << "number of mixture components = " << nbMixtureComponents << endl;
        cout << "covariance offset = " << covarianceOffset << endl;
        cout << "mixture mixtureCoeffs:\n";
        for (int c=0; c<nbMixtureComponents; c++)
            cout << mixtureCoeffs[c] << " ";
        cout << "\n";
        cout << "means:\n";
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d=0; d<dimension; d++) {
                cout << meanOfComponent(c)[d] << " ";
            }
            cout << "\n";
        }
        cout << "\n";
        cout << "covariances:\n";
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d1=0; d1<dimension; d1++) {
                for (int d2=0; d2<dimension; d2++) {
                    cout << covarianceOfComponent(c)[d1*dimension+d2] << " ";
                }
                cout << "\n";
            }
            cout << "\n";
        }
        cout << "\n";
    }
    
#pragma mark -
#pragma mark Python
    /*! @name Python methods */
#ifdef SWIGPYTHON
    double play(int dimension_, double *observation,
                int nbMixtureComponents_, double *beta_)
    {
        float *obs_float = new float[dimension_];
        for (int i=0 ; i < dimension_ ; i++)
            obs_float[i] = float(observation[i]);
        
        double likelihood = play(obs_float);
        
        for (int i=0; i<nbMixtureComponents_; i++) {
            beta_[i] = beta[i];
        }
        
        delete[] obs_float;
        
        return likelihood;
    }
    
#endif
    
#pragma mark -
#pragma mark Training algorithm
    /*! @name Training Algorithm */
    virtual void initTraining()
    {
        initParametersToDefault();
        initMeansWithFirstPhrase();
        updateInverseCovariances();
    }
    
    virtual void initParametersToDefault()
    {
        double norm_coeffs(0.);
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d=0; d<dimension; d++) {
                for (int d2=0; d2<dimension; d2++) {
                    covariance[c*dimension*dimension+d*dimension+d2] = 1.;
                }
                covariance[c*dimension*dimension+d*dimension+d] += covarianceOffset;
            }
            mixtureCoeffs[c] = 1./float(nbMixtureComponents);
            norm_coeffs += mixtureCoeffs[c];
        }
        for (int c=0; c<nbMixtureComponents; c++) {
            mixtureCoeffs[c] /= norm_coeffs;
        }
    }
    
    virtual void finishTraining()
    {}
    
    double train_EM_update()
    {
        double log_prob(0.);
        
        int totalLength(0);
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
            totalLength += it->second->getlength();
        
        vector< vector<double> > p(nbMixtureComponents);
        vector<double> E(nbMixtureComponents, 0.0);
        for (int c=0; c<nbMixtureComponents; c++) {
            p[c].resize(totalLength);
            E[c] = 0.;
        }
        
        int tbase(0);
        
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++) {
            int T = it->second->getlength();
            for (int t=0; t<T; t++) {
                double norm_const(0.);
                for (int c=0; c<nbMixtureComponents; c++)
                {
                    p[c][tbase+t] = obsProb(it->second->get_dataPointer(t), c);
                    
                    if (p[c][tbase+t] == 0. || isnan(p[c][tbase+t]) || isinf(p[c][tbase+t])) {
                        p[c][tbase+t] = 1e-100;
                    }
                    norm_const += p[c][tbase+t];
                }
                for (int c=0; c<nbMixtureComponents; c++) {
                    p[c][tbase+t] /= norm_const;
                    E[c] += p[c][tbase+t];
                }
                if (norm_const > 1.) cout << "Training Error: covarianceOffset is too small\n";//throw runtime_error("Training Error: covarianceOffset is too small");
                log_prob += log(norm_const);
            }
            tbase += T;
        }
        
        // Estimate Mixture coefficients
        for (int c=0; c<nbMixtureComponents; c++) {
            mixtureCoeffs[c] = E[c]/double(totalLength);
        }
        
        // Estimate means
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d=0; d<dimension; d++) {
                meanOfComponent(c)[d] = 0.;
                tbase = 0;
                for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++) {
                    int T = it->second->getlength();
                    for (int t=0; t<T; t++) {
                        meanOfComponent(c)[d] += p[c][tbase+t] * (*it->second)(t, d);
                    }
                    tbase += T;
                }
                meanOfComponent(c)[d] /= E[c];
            }
        }
        
        //estimate covariances
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d1=0; d1<dimension; d1++) {
                for (int d2=0; d2<dimension; d2++) {
                    covarianceOfComponent(c)[d1*dimension+d2] = 0.;
                    tbase = 0;
                    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++) {
                        int T = it->second->getlength();
                        for (int t=0; t<T; t++) {
                            covarianceOfComponent(c)[d1*dimension+d2] += p[c][tbase+t]
                            * ((*it->second)(t, d1) - meanOfComponent(c)[d1])
                            * ((*it->second)(t, d2) - meanOfComponent(c)[d2]);
                        }
                        tbase += T;
                    }
                    covarianceOfComponent(c)[d1*dimension+d2] /= E[c];
                }
            }
        }
        
        addCovarianceOffset();
        updateInverseCovariances();
        
        return log_prob;
    }
    
#pragma mark -
#pragma mark Protected attributes
    /*! @name Protected attributes */
protected:
    int dimension;
    
    int nbMixtureComponents;
    float covarianceOffset;
    
    vector<float> inverseCovariance;
    vector<double> covarianceDeterminant;
    
    vector<double> beta;
};


#endif
