//
//  kmeans.cpp
//  mhmm
//
//  Created by Jules Francoise on 27/09/2014.
//
//

#include "kmeans.h"
#include <limits>

#pragma mark -
#pragma mark === Public Interface ===
#pragma mark > Constructors
KMeans::KMeans(TrainingSet *trainingSet,
               int nbClusters)
: trainingSet(trainingSet),
  trainingInitType(RANDOM),
  nbClusters_(nbClusters > 0 ? nbClusters : 1),
  dimension_(1),
  is_training_(false),
  training_maxIterations_(KMEANS_DEFAULT_MAX_ITERATIONS),
  training_relativeDistanceThreshold_(KMEANS_DEFAULT_RELATIVE_VARIATION_THRESHOLD),
  trainingCallbackFunction_(NULL)
{
    if (this->trainingSet)
        dimension_ = this->trainingSet->dimension();
    centers.resize(nbClusters*dimension_, 0.0);
}

KMeans::KMeans(KMeans const& src)
{
    this->nbClusters_ = src.nbClusters_;
    this->dimension_ = src.dimension_;
    this->centers = src.centers;
}

KMeans& KMeans::operator=(KMeans const& src)
{
    if(this != &src)
    {
        this->nbClusters_ = src.nbClusters_;
        this->dimension_ = src.dimension_;
        this->centers = src.centers;
    }
    return *this;
}

KMeans::~KMeans()
{
    centers.clear();
}

void KMeans::notify(string attribute)
{
    if (!trainingSet) return;
    if (attribute == "dimension") {
        dimension_ = trainingSet->dimension();
        return;
    }
    if (attribute == "destruction") {
        trainingSet = NULL;
        return;
    }
}

#pragma mark > Accessors
#pragma mark -
#pragma mark Accessors
bool KMeans::is_training() const
{
    return is_training_;
}

void KMeans::set_trainingSet(TrainingSet *trainingSet)
{
    PREVENT_ATTR_CHANGE();
    this->trainingSet = trainingSet;
}

unsigned int KMeans::get_nbClusters()
{
    return nbClusters_;
}

void KMeans::set_nbClusters(unsigned int nbClusters)
{
    nbClusters_ = nbClusters;
    centers.resize(nbClusters * dimension_, 0.0);
}

unsigned int KMeans::get_training_maxIterations() const
{
    return training_maxIterations_;
}

void KMeans::set_training_maxIterations(unsigned int maxIterations)
{
    training_maxIterations_ = maxIterations;
}

unsigned int KMeans::get_training_relativeDistanceThreshold() const
{
    return training_relativeDistanceThreshold_;
}

void KMeans::set_training_relativeDistanceThreshold(float threshold)
{
    training_relativeDistanceThreshold_ = threshold;
}

unsigned int KMeans::dimension() const
{
    return dimension_;
}

#pragma mark > Training
void KMeans::train()
{
#ifdef USE_PTHREAD
    pthread_mutex_lock(&trainingMutex);
#endif
    if (!this->trainingSet || this->trainingSet->is_empty())
    {
#ifdef USE_PTHREAD
        pthread_mutex_unlock(&trainingMutex);
        if (this->trainingCallbackFunction_) {
            pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
            this->trainingCallbackFunction_(this, TRAINING_ERROR, this->trainingExtradata_);
            pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
            pthread_testcancel();
        }
#else
        if (this->trainingCallbackFunction_) {
            this->trainingCallbackFunction_(this, TRAINING_ERROR, this->trainingExtradata_);
        }
#endif
        return;
    }
    
    is_training_ = true;
    
    dimension_ = trainingSet->dimension();
    centers.resize(nbClusters_ * dimension_, 0.0);
    if (trainingInitType == RANDOM)
        randomizeClusters();
    else
        initClustersWithFirstPhrase();
    
    //    for (vector<float>::iterator it=centers.begin(); it!=centers.end(); ++it) {
    //        cout << *it << " ";
    //    }
    //    cout << endl;
    
    trainingNbIterations = 0;
    for (trainingNbIterations=0; trainingNbIterations<training_maxIterations_; ++trainingNbIterations) {
        vector<float> previous_centers = centers;
        
        updateCenters(previous_centers);
        
        //        for (vector<float>::iterator it=centers.begin(); it!=centers.end(); ++it) {
        //            cout << *it << " ";
        //        }
        //        cout << endl;
        
        float meanClusterDistance(0.0);
        float maxRelativeCenterVariation(0.0);
        for (unsigned int k=0; k<nbClusters_; ++k) {
            for (unsigned int l=0; l<nbClusters_; ++l) {
                if (k != l) {
                    meanClusterDistance += euclidian_distance(&centers[k*dimension_], &centers[l*dimension_], dimension_);
                }
            }
            maxRelativeCenterVariation = max(euclidian_distance(&previous_centers[k*dimension_], &centers[k*dimension_], dimension_),
                                             maxRelativeCenterVariation);
        }
        meanClusterDistance /= float(nbClusters_ * (nbClusters_ - 1));
        maxRelativeCenterVariation /= float(nbClusters_);
        maxRelativeCenterVariation /= meanClusterDistance;
        if (maxRelativeCenterVariation < training_relativeDistanceThreshold_)
            break;
        
#ifdef USE_PTHREAD
        if (this->trainingCallbackFunction_) {
            pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
            this->trainingCallbackFunction_(this, TRAINING_RUN, this->trainingExtradata_);
            pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
            pthread_testcancel();
        }
#else
        if (this->trainingCallbackFunction_) {
            this->trainingCallbackFunction_(this, TRAINING_RUN, this->trainingExtradata_);
        }
#endif
    }
    
#ifdef USE_PTHREAD
    if (trainingCallbackFunction_) {
        pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
        trainingCallbackFunction_(this, TRAINING_DONE, trainingExtradata_);
        pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
        pthread_testcancel();
    }
#else
    if (trainingCallbackFunction_) {
        trainingCallbackFunction_(this, TRAINING_DONE, trainingExtradata_);
    }
#endif
    
#ifdef USE_PTHREAD
    pthread_mutex_unlock(&trainingMutex);
#endif
}

void KMeans::initClustersWithFirstPhrase()
{
    if (!this->trainingSet || this->trainingSet->is_empty())
        return;
    int step = this->trainingSet->begin()->second->length() / nbClusters_;
    
    int offset(0);
    for (int c=0; c<nbClusters_; c++) {
        for (int d=0; d<dimension_; d++) {
            centers[c*dimension_+d] = 0.0;
        }
        for (int t=0; t<step; t++) {
            for (int d=0; d<dimension_; d++) {
                centers[c*dimension_+d] += (*this->trainingSet->begin()->second)(offset+t, d) / float(step);
            }
        }
        offset += step;
    }
}


void KMeans::randomizeClusters()
{
    vector<float> trainingSetVariance(dimension_, 1.);
    if (trainingSet)
        vector<float> trainingSetVariance = trainingSet->variance();
    for (unsigned int k=0; k<nbClusters_; ++k) {
        for (unsigned int d=0; d<dimension_; ++d) {
            centers[k*dimension_ + d] = trainingSetVariance[d] * (2. * random() / float(RAND_MAX) - 1.);
        }
    }
}

void KMeans::updateCenters(vector<float>& previous_centers)
{
    unsigned int phraseIndex(0);
    centers.assign(nbClusters_*dimension_, 0.0);
    vector<unsigned int> numFramesPerCluster(nbClusters_, 0);
    for (TrainingSet::phrase_iterator it=trainingSet->begin(); it!=trainingSet->end(); ++it, ++phraseIndex) {
        for (unsigned int t=0; t<it->second->length(); ++t) {
            float min_distance;
            if (trainingSet->is_bimodal()) {
                vector<float> frame(dimension_);
                for (unsigned int d=0; d<dimension_; ++d) {
                    frame[d] = it->second->at(t, d);
                }
                min_distance = euclidian_distance(&frame[0],
                                                  &previous_centers[0],
                                                  dimension_);
            } else {
                min_distance = euclidian_distance(it->second->get_dataPointer(t),
                                                  &previous_centers[0],
                                                  dimension_);
            }
            int cluster_membership(0);
            for (unsigned int k=1; k<nbClusters_; ++k) {
                float distance;
                if (trainingSet->is_bimodal()) {
                    vector<float> frame(dimension_);
                    for (unsigned int d=0; d<dimension_; ++d) {
                        frame[d] = it->second->at(t, d);
                    }
                    distance = euclidian_distance(&frame[0],
                                                  &previous_centers[k*dimension_],
                                                  dimension_);
                } else {
                    distance = euclidian_distance(it->second->get_dataPointer(t),
                                                  &previous_centers[k*dimension_],
                                                  dimension_);
                }
                if (distance < min_distance)
                {
                    cluster_membership = k;
                    min_distance = distance;
                }
            }
            numFramesPerCluster[cluster_membership]++;
            for (unsigned int d=0; d<dimension_; ++d) {
                centers[cluster_membership * dimension_ + d] += it->second->at(t, d);
            }
        }
    }
    for (unsigned int k=0; k<nbClusters_; ++k) {
        if (numFramesPerCluster[k] > 0)
            for (unsigned int d=0; d<dimension_; ++d) {
                centers[k * dimension_ + d] /= float(numFramesPerCluster[k]);
            }
    }
}

void* KMeans::train_func(void *context)
{
    ((KMeans *)context)->train();
    return NULL;
}

#ifdef USE_PTHREAD
void KMeans::abortTraining(pthread_t this_thread)
{
    if (!is_training_)
        return;
    pthread_cancel(this_thread);
    void *status;
    pthread_join(this_thread, &status);
    pthread_mutex_unlock(&trainingMutex);
    is_training_ = false;
    if (trainingCallbackFunction_) {
        trainingCallbackFunction_(this, TRAINING_ABORT, trainingExtradata_);
    }
}
#endif

void KMeans::set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata)
{
    PREVENT_ATTR_CHANGE();
    trainingExtradata_ = extradata;
    trainingCallbackFunction_ = callback;
}

#pragma mark > Performance
void KMeans::performance_init()
{
    results_distances.resize(nbClusters_, 0.0);
}

void KMeans::performance_update(vector<float> const& observation)
{
    if (observation.size() != dimension_)
        throw runtime_error("Dimensions Don't Agree");
    results_likeliest = 0;
    float minDistance(numeric_limits<float>::max());
    for (unsigned int k=0; k<nbClusters_; ++k) {
        results_distances[k] = euclidian_distance(&observation[0], &centers[k*dimension_], dimension_);
        if (results_distances[k] < minDistance) {
            minDistance = results_distances[k];
            results_likeliest = k;
        }
    }
}

#pragma mark -
#pragma mark File IO
JSONNode KMeans::to_json() const
{
    JSONNode json_model(JSON_NODE);
    json_model.set_name("KMeans");
    
    json_model.push_back(JSONNode("dimension", dimension_));
    json_model.push_back(JSONNode("nbclusters", nbClusters_));
    json_model.push_back(vector2json(centers, "centers"));
    
    return json_model;
}

void KMeans::from_json(JSONNode root)
{
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root.name());
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Dimension
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "dimension")
            throw JSONException("Wrong name: was expecting 'dimension'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        dimension_ = static_cast<unsigned int>(root_it->as_int());
        ++root_it;
        
        // Get Number of Clusters
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "nbclusters")
            throw JSONException("Wrong name: was expecting 'nbclusters'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        nbClusters_ = static_cast<unsigned int>(root_it->as_int());
        ++root_it;
        
        // Get Mixture Coefficients
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "centers")
            throw JSONException("Wrong name: was expecting 'centers'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        json2vector(*root_it, centers, nbClusters_);
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}

# pragma mark > Utility
template <typename numType>
numType euclidian_distance(const numType* vector1,
                           const numType* vector2,
                           unsigned int dimension) {
    numType distance(0.0);
    for (unsigned int d=0; d<dimension; d++) {
        distance += pow(vector1[d] - vector2[d], 2);
    }
    return sqrt(distance);
}

