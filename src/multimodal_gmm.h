//
//  mgmm.h
//  mhmm
//
//  Created by Jules Francoise on 16/11/12.
//
//

#ifndef __mhmm__mgmm__
#define __mhmm__mgmm__

#include "em_based_learning_model.h"
#include "matrix.h"

#define MGMM_DEFAULT_NB_MIXTURE_COMPONENTS 1
#define MGMM_DEFAULT_COVARIANCE_OFFSET 0.01

/*!
 * @class MultimodalGMM
 * @brief Multimodal Gaussian Mixture Model
 *
 * Can be either autonomous or a state of the MHMM: defines observation probabilities for each state
 @todo detail documentation
 @tparam ownData defines if phrases has own data or shared memory
 */
template<bool ownData>
class MultimodalGMM : public EMBasedLearningModel< GestureSoundPhrase<ownData>, int> {
public:
    typedef typename map<int, GestureSoundPhrase<ownData>* >::iterator phrase_iterator;
    
    vector<float> mean;
    vector<float> covariance;
    vector<float> mixtureCoeffs;
    
    vector<float> observation_sound;
    vector<float> covariance_sound;
    double covariance_sound_det;
    
#pragma mark -
#pragma mark Constructors
    /*! @name Constructors */
    /*!
     Constructor
     @param \_trainingSet training set associated with the model
     @param nbMixtureComponents\_ number of mixture components
     @param covarianceOffset_ offset to add to the diagonal of covariances matrices (useful to guarantee convergence)
     */
    MultimodalGMM(TrainingSet<GestureSoundPhrase<ownData>, int> *_trainingSet=NULL,
                  int nbMixtureComponents_ = MGMM_DEFAULT_NB_MIXTURE_COMPONENTS,
                  float covarianceOffset_= MGMM_DEFAULT_COVARIANCE_OFFSET)
    : EMBasedLearningModel<GestureSoundPhrase<ownData>, int>(_trainingSet)
    {
        nbMixtureComponents  = nbMixtureComponents_;
        covarianceOffset     = covarianceOffset_;
        
        if (this->trainingSet) {
            dimension_gesture = this->trainingSet->get_dimension_gesture();
            dimension_sound = this->trainingSet->get_dimension_sound();
        } else {
            dimension_gesture = 1;
            dimension_sound = 1;
        }
        dimension_total = dimension_gesture + dimension_sound;
        
        reallocParameters();
        initTraining();
    }
    
    /*!
     Copy constructor
     */
    MultimodalGMM(MultimodalGMM const& src) : EMBasedLearningModel< GestureSoundPhrase<ownData>, int>(src)
    {
        copy(this, src);
    }
    
    /*!
     Assignment
     */
    MultimodalGMM& operator=(MultimodalGMM const& src)
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
    void copy(MultimodalGMM *dst, MultimodalGMM const& src)
    {
        EMBasedLearningModel<GestureSoundPhrase<ownData>, int>::copy(dst, src);
        dst->nbMixtureComponents     = src.nbMixtureComponents;
        dst->covarianceOffset        = src.covarianceOffset;
        dst->covarianceDeterminant   = src.covarianceDeterminant;
        
        dst->dimension_gesture = src.dimension_gesture;
        dst->dimension_sound = src.dimension_sound;
        dst->dimension_total = dst->dimension_gesture + dst->dimension_sound;
        
        dst->mixtureCoeffs = src.mixtureCoeffs;
        dst->mean = src.mean;
        dst->covariance = src.covariance;
        dst->inverseCovariance = src.inverseCovariance;
        dst->inverseCovariance_gesture = src.inverseCovariance_gesture;
        dst->covarianceDeterminant = src.covarianceDeterminant;
        dst->covarianceDeterminant_gesture = src.covarianceDeterminant_gesture;
        
        dst->reallocParameters();
    }
    
    /*!
     Destructor
     */
    ~MultimodalGMM()
    {
        mean.clear();
        covariance.clear();
        inverseCovariance.clear();
        inverseCovariance_gesture.clear();
        covarianceDeterminant.clear();
        covarianceDeterminant_gesture.clear();
        mixtureCoeffs.clear();
    }
    
    /*!
     handle notifications of the training set
     
     here only the dimensions attributes of the training set are considered
     */
    void notify(string attribute)
    {
        if (!this->trainingSet) return;
        if (attribute == "dimension_gesture") {
            dimension_gesture = this->trainingSet->get_dimension_gesture();
            dimension_total = dimension_gesture + dimension_sound;
            reallocParameters();
            return;
        }
        if (attribute == "dimension_sound") {
            dimension_sound = this->trainingSet->get_dimension_sound();
            dimension_total = dimension_gesture + dimension_sound;
            reallocParameters();
            return;
        }
    }
    
    /*!
     Set training set associated with the model
     */
    void set_trainingSet(TrainingSet<GestureSoundPhrase<ownData>, int> *_trainingSet)
    {
        this->trainingSet = _trainingSet;
        if (this->trainingSet) {
            dimension_gesture = this->trainingSet->get_dimension_gesture();
            dimension_sound = this->trainingSet->get_dimension_sound();
        } else {
            dimension_gesture = 0;
            dimension_sound = 0;
        }
        dimension_total = dimension_gesture + dimension_sound;
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
        mean.resize(nbMixtureComponents*dimension_total);
        covariance.resize(nbMixtureComponents*dimension_total*dimension_total);
        inverseCovariance.resize(nbMixtureComponents*dimension_total*dimension_total);
        inverseCovariance_gesture.resize(nbMixtureComponents*dimension_gesture*dimension_gesture);
        mixtureCoeffs.resize(nbMixtureComponents);
        covarianceDeterminant.resize(nbMixtureComponents);
        covarianceDeterminant_gesture.resize(nbMixtureComponents);
        observation_sound.resize(dimension_sound);
        covariance_sound.resize(dimension_sound*dimension_sound);
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
            for (int d=0; d<dimension_total; d++) {
                meanOfComponent(c)[d] = 0.0;
            }
            for (int t=0; t<step; t++) {
                for (int d=0; d<dimension_total; d++) {
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
            for (int d=0; d<dimension_total; d++) {
                //mean[c*dimension_total+d] = 0.;
                for (int d2=0; d2<dimension_total; d2++) {
                    covariance[c*dimension_total*dimension_total+d*dimension_total+d2] = 0.;
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
            for (int d=0; d<dimension_total; d++)
                covarianceOfComponent(c)[d*dimension_total+d] += covarianceOffset;
        }
    }
    
#pragma mark -
#pragma mark Accessors
    /*! @name Accessors */
    int get_dimension_gesture() const
    {
        return dimension_gesture;
    }
    
    int get_dimension_sound() const
    {
        return dimension_sound;
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
     Multimodal Gesture-Sound Observation probability
     @param obs_gesture gesture observation vector
     @param obs_sound sound observation vector
     @param mixtureComponent index of the mixture component. if unspecified or negative, 
     full mixture observation probability is computed
     */
    double obsProb_gestureSound(const float *obs_gesture, const float *obs_sound, int mixtureComponent=-1)
    {
        double p(0.);
        
        if (mixtureComponent < 0) {
            for (mixtureComponent=0; mixtureComponent<nbMixtureComponents; mixtureComponent++) {
                p += obsProb_gestureSound(obs_gesture, obs_sound, mixtureComponent);
            }
        } else {
            p = mixtureCoeffs[mixtureComponent] * gaussianProbabilityFullCovariance_GestureSound(obs_gesture,
                                                                                                 obs_sound,
                                                                                                 meanOfComponent(mixtureComponent),
                                                                                                 covarianceDeterminant[mixtureComponent],
                                                                                                 inverseCovarianceOfComponent(mixtureComponent),
                                                                                                 dimension_gesture,
                                                                                                 dimension_sound);
        }
        
        return p;
    }
    
    /*!
     Unimodal Gesture Observation probability
     @param obs_gesture gesture observation vector
     @param mixtureComponent index of the mixture component. if unspecified or negative,
     full mixture observation probability is computed
     */
    double obsProb_gesture(const float *obs, int mixtureComponent = -1)
    {
        double p(0.);
        
        if (mixtureComponent < 0)
        {
            for (mixtureComponent=0; mixtureComponent<nbMixtureComponents; mixtureComponent++) {
                p += obsProb_gesture(obs, mixtureComponent);
            }
        }
        else
        {
            p = mixtureCoeffs[mixtureComponent] * gaussianProbabilityFullCovariance(obs,
                                                                                    meanOfComponent(mixtureComponent),
                                                                                    covarianceDeterminant_gesture[mixtureComponent],
                                                                                    inverseCovariance_gesture_OfComponent(mixtureComponent),
                                                                                    dimension_gesture);
        }
        
        return p;
    }
    
    /*!
     Unimodal Sound Observation probability
     @param obs_sound sound observation vector
     @param mixtureComponent index of the mixture component. if unspecified or negative,
     full mixture observation probability is computed
     */
    double obsProb_sound(const float *obs, int mixtureComponent = -1)
    {
        double p(0.);
        
        if (mixtureComponent < 0) {
            for (mixtureComponent=0; mixtureComponent<nbMixtureComponents; mixtureComponent++) {
                p += obsProb_sound(obs, mixtureComponent);
            }
        } else {
            double det;
            vector<float> covSndArray = createCovariance_sound_OfComponent(mixtureComponent);
            Matrix<float> covSndMatrix(dimension_sound, dimension_sound, covSndArray.begin());
            Matrix<float> inverseCovSnd = covSndMatrix.pinv(&det);
            
            p = mixtureCoeffs[mixtureComponent] * gaussianProbabilityFullCovariance(obs,
                                                                                    mean_sound_OfComponent(mixtureComponent),
                                                                                    det,
                                                                                    inverseCovSnd.data,
                                                                                    dimension_sound);
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
        return mean.begin() + component * dimension_total;
    }
    
    /*!
     get mean of a particular gaussian component (sound)
     */
    vector<float>::iterator mean_sound_OfComponent(int component)
    {
        return mean.begin() + component * dimension_total + dimension_gesture;
    }
    
    /*!
     get covariance of a particular gaussian component (multimodal)
     */
    vector<float>::iterator covarianceOfComponent(int component)
    {
        return covariance.begin() + component * dimension_total * dimension_total;
    }
    
    /*!
     get inverse covariance of a particular gaussian component (multimodal)
     */
    vector<float>::iterator inverseCovarianceOfComponent(int component)
    {
        return inverseCovariance.begin() + component * dimension_total * dimension_total;
    }
    
    /*!
     get inverse covariance of a particular gaussian component (gesture)
     */
    vector<float>::iterator inverseCovariance_gesture_OfComponent(int component)
    {
        return inverseCovariance_gesture.begin() + component * dimension_gesture * dimension_gesture;
    }
    
    /*!
     Create the unimodal covariance matrix of the gesture modality for a particular component
     */
    vector<float> createCovariance_gesture_OfComponent(int component)
    {
        vector<float> cov_gesture(dimension_gesture*dimension_gesture);
        for (int d1=0; d1<dimension_gesture; d1++) {
            for (int d2=0; d2<dimension_gesture; d2++) {
                cov_gesture[d1*dimension_gesture+d2] = covarianceOfComponent(component)[d1*dimension_total+d2];
            }
        }
        return cov_gesture;
    }
    
    /*!
     Create the unimodal covariance matrix of the sound modality for a particular component
     */
    vector<float> createCovariance_sound_OfComponent(int component)
    {
        vector<float> cov_sound(dimension_sound*dimension_sound);
        for (int d1=0; d1<dimension_sound; d1++) {
            for (int d2=0; d2<dimension_sound; d2++) {
                cov_sound[d1*dimension_sound+d2] = covarianceOfComponent(component)[d1*dimension_total+dimension_gesture+d2];
            }
        }
        return cov_sound;
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
        double maxProb = obsProb_gesture(obs, component);
        double prob;
        for (int c=1 ; c<nbMixtureComponents; c++) {
            prob = obsProb_gesture(obs, c);
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
        Matrix<float> cov_matrix(dimension_total, dimension_total, false);
        Matrix<float> cov_matrix_gesture(dimension_gesture, dimension_gesture, true);
        
        Matrix<float> *inverseMat;
        double det;
        
        for (int c=0; c<nbMixtureComponents; c++)
        {
            // Update inverse covariance for gesture+sound
            cov_matrix.data = covarianceOfComponent(c);
            inverseMat = cov_matrix.pinv(&det);
            covarianceDeterminant[c] = det;
            vectorCopy(inverseCovarianceOfComponent(c), inverseMat->data, dimension_total*dimension_total);
            delete inverseMat;
            
            // Update inverse covariance for gesture only
            cov_matrix_gesture._data = createCovariance_gesture_OfComponent(c);
            inverseMat = cov_matrix_gesture.pinv(&det);
            covarianceDeterminant_gesture[c] = det;
            vectorCopy(inverseCovariance_gesture_OfComponent(c), inverseMat->data, dimension_gesture*dimension_gesture);
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
    double regression(float *obs)
    {
        vector<float>  newMeanSound = regression_evaluateMeanSound(obs);
        regression_beta(obs);
        
        for (int d=0; d<dimension_sound; d++) {
            obs[dimension_gesture+d] = 0.;
            for (int k=0; k<nbMixtureComponents; k++) {
                obs[dimension_gesture+d] += beta[k] * newMeanSound[k*dimension_sound+d];
            }
        }
        
        for (int i=0; i<dimension_sound; i++)
            observation_sound[i] = obs[dimension_gesture+i];
        regression_estimateCovariance();
        
        return obsProb_gesture(obs, -1);
    }
    
    /*!
     estimate the sound parameters for each component from the gesture observation vector
     */
    vector<float> regression_evaluateMeanSound(const float *obs)
    {
        vector<float> newMeanSound(nbMixtureComponents*dimension_sound);
        float tmp;
        
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d=0; d<dimension_sound; d++) {
                newMeanSound[c*dimension_sound+d] = mean_sound_OfComponent(c)[d];
                for (int e=0; e<dimension_gesture; e++) {
                    tmp = 0.;
                    for (int f=0; f<dimension_gesture; f++) {
                        tmp += inverseCovariance_gesture_OfComponent(c)[e*dimension_gesture+f] * (obs[f] - meanOfComponent(c)[f]);
                    }
                    newMeanSound[c*dimension_sound+d] += covariance_sg(c, d, e) * tmp;
                }
            }
        }
        return newMeanSound;
    }
    
    /*!
     get cross-modal sound-gesture covariance
     @param component gaussian mixture component
     @param d1 first dimension
     @param d2 second dimension
     */
    float covariance_sg(int component, int d1, int d2)
    {
        return covarianceOfComponent(component)[(d1+dimension_gesture)*dimension_total + d2];
    }
    
    /*!
     Compute likelihoods of each components given a gesture observation vector
     */
    void regression_beta(const float *obs)
    {
        double norm_const(0.);
        for (int c=0; c<nbMixtureComponents; c++) {
            beta[c] = obsProb_gesture(obs, c);
            norm_const += beta[c];
        }
        for (int c=0; c<nbMixtureComponents; c++) {
            beta[c] /= norm_const;
        }
    }
    
    /*!
     estimate the covariance on sound parameters
     */
    void regression_estimateCovariance()//, float *estimatedCovariance)
    {
        for (int d1=0; d1<dimension_sound; d1++) {
            for (int d2=0; d2<dimension_sound; d2++) {
                covariance_sound[d1*dimension_sound+d2] = 0.0;
                for (int c=0; c<nbMixtureComponents; c++) {
                    covariance_sound[d1*dimension_sound+d2] += beta[c] * beta[c] * conditionalSoundCovariance[c*dimension_sound*dimension_sound + d1*dimension_sound + d2];
                }
            }
        }
        
        // Compute determinant
        Matrix<float> cov_matrix(dimension_sound, dimension_sound, false);
        Matrix<float> *inverseMat;
        double det;
        cov_matrix.data = covariance_sound.begin();
        inverseMat = cov_matrix.pinv(&det);
        covariance_sound_det = det;
        delete inverseMat;
    }
    
    void estimateConditionalSoundCovariance()
    {
        conditionalSoundCovariance.resize(nbMixtureComponents*dimension_sound*dimension_sound);
        vector<float>::iterator currentCovPoint = conditionalSoundCovariance.begin();
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d1=0; d1<dimension_sound; d1++) {
                for (int d2=0; d2<dimension_sound; d2++) {
                    *currentCovPoint = covarianceOfComponent(c)[(d1+dimension_gesture)*dimension_total + dimension_gesture + d2];
                    for (int i=0; i<dimension_gesture; i++) {
                        double sum1(0.);
                        for (int j=0; j<dimension_gesture; j++) {
                            sum1 += inverseCovariance_gesture_OfComponent(c)[i*dimension_gesture + j] * covariance_gs(c, j, d2);
                        }
                        *currentCovPoint -= covariance_sg(c, d1, i) * sum1;
                    }
                    currentCovPoint++;
                }
            }
        }
    }
    
    float covariance_gs(int component, int d1, int d2)
    {
        return covarianceOfComponent(component)[d1*dimension_total + dimension_gesture + d2];
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
        double prob = regression(obs);
        return this->updateLikelihoodBuffer(prob);
    }
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    void write(ostream& outStream, bool writeTrainingSet=false)
    {
        outStream << "# Multimodal GMM \n";
        outStream << "# =========================================\n";
        EMBasedLearningModel<GestureSoundPhrase<ownData>, int>::write(outStream, writeTrainingSet);
        outStream << "# Dimensions\n";
        outStream << dimension_gesture << " " << dimension_sound << endl;
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
            for (int d=0; d<dimension_total; d++) {
                outStream << meanOfComponent(c)[d] << " ";
            }
            outStream << endl;
        }
        outStream << "# Covariance\n";
        for (int c=0 ; c<nbMixtureComponents; c++) {
            for (int d1=0; d1<dimension_total; d1++) {
                for (int d2=0; d2<dimension_total; d2++) {
                    outStream << covarianceOfComponent(c)[d1*dimension_total+d2] << " ";
                }
                outStream << endl;
            }
        }
    }
    
    void read(istream& inStream, bool readTrainingSet=false)
    {
        EMBasedLearningModel<GestureSoundPhrase<ownData>, int>::read(inStream, readTrainingSet);
        
        // Get Dimensions
        skipComments(&inStream);
        inStream >> dimension_gesture;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);;
        inStream >> dimension_sound;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);;
        
        dimension_total = dimension_gesture + dimension_sound;
        
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
            for (int d=0; d<dimension_total; d++) {
                inStream >> meanOfComponent(c)[d];
                if (!inStream.good())
                    throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);;
            }
        }
        
        // Get Covariance
        skipComments(&inStream);
        for (int c=0 ; c<nbMixtureComponents; c++) {
            for (int d1=0; d1<dimension_total; d1++) {
                for (int d2=0; d2<dimension_total; d2++) {
                    inStream >> covarianceOfComponent(c)[d1*dimension_total+d2];
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
        
        cout << "Dimensions = " << dimension_gesture << " " << dimension_sound << endl;
        cout << "number of mixture components = " << nbMixtureComponents << endl;
        cout << "covariance offset = " << covarianceOffset << endl;
        cout << "mixture mixtureCoeffs:\n";
        for (int c=0; c<nbMixtureComponents; c++)
            cout << mixtureCoeffs[c] << " ";
        cout << "\n";
        cout << "means:\n";
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d=0; d<dimension_gesture+dimension_sound; d++) {
                cout << meanOfComponent(c)[d] << " ";
            }
            cout << "\n";
        }
        cout << "\n";
        cout << "covariances:\n";
        for (int c=0; c<nbMixtureComponents; c++) {
            for (int d1=0; d1<dimension_gesture+dimension_sound; d1++) {
                for (int d2=0; d2<dimension_gesture+dimension_sound; d2++) {
                    cout << covarianceOfComponent(c)[d1*dimension_total+d2] << " ";
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
    double play(int dimension_gesture_, double *observation_gesture,
                int dimension_sound_, double *observation_sound_out,
                int nbMixtureComponents_, double *beta_,
                int dimension_sound_square, double *outCovariance)
    {
        float *observation_total = new float[dimension_total];
        for (int d=0; d<dimension_gesture; d++) {
            observation_total[d] = float(observation_gesture[d]);
        }
        for (int d=0; d<dimension_sound; d++)
            observation_total[d+dimension_gesture] = 0.;
        
        double likelihood = play(observation_total);
        
        for (int d=0; d<dimension_sound_; d++) {
            observation_sound_out[d] = double(observation_total[dimension_gesture+d]);
        }
        delete[] observation_total;
        
        for (int i=0; i<nbMixtureComponents_; i++) {
            beta_[i] = beta[i];
        }
        
        for (int d=0; d<dimension_sound_square; d++) {
            outCovariance[d] = covariance_sound[d];
        }
        
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
            for (int d=0; d<dimension_total; d++) {
                // mean[c*dimension_total+d] = float(rand()%100)/100.;
                for (int d2=0; d2<dimension_total; d2++) {
                    covariance[c*dimension_total*dimension_total+d*dimension_total+d2] = 1.;
                }
                covariance[c*dimension_total*dimension_total+d*dimension_total+d] += covarianceOffset;
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
                for (int c=0; c<nbMixtureComponents; c++) {
                    p[c][tbase+t] = obsProb_gestureSound(it->second->get_dataPointer_gesture(t),
                                                         it->second->get_dataPointer_sound(t),
                                                         c);
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
            for (int d=0; d<dimension_total; d++) {
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
            for (int d1=0; d1<dimension_total; d1++) {
                for (int d2=0; d2<dimension_total; d2++) {
                    covarianceOfComponent(c)[d1*dimension_total+d2] = 0.;
                    tbase = 0;
                    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++) {
                        int T = it->second->getlength();
                        for (int t=0; t<T; t++) {
                            covarianceOfComponent(c)[d1*dimension_total+d2] += p[c][tbase+t]
                            * ((*it->second)(t, d1) - meanOfComponent(c)[d1])
                            * ((*it->second)(t, d2) - meanOfComponent(c)[d2]);
                        }
                        tbase += T;
                    }
                    covarianceOfComponent(c)[d1*dimension_total+d2] /= E[c];
                }
            }
        }
        
        addCovarianceOffset();
        updateInverseCovariances();
        estimateConditionalSoundCovariance();
        
        return log_prob;
    }
    
#pragma mark -
#pragma mark Protected attributes
    /*! @name Protected attributes */
protected:
    int dimension_gesture;
    int dimension_sound;
    int dimension_total;
    
    int nbMixtureComponents;
    float covarianceOffset;
    
    vector<float> inverseCovariance;
    vector<float> inverseCovariance_gesture;
    vector<double> covarianceDeterminant;
    vector<double> covarianceDeterminant_gesture;
    
    vector<float> conditionalSoundCovariance;
    vector<double> beta;
};

#endif
