//
//  training_set.h
//  rtml
//
//  Created by Jules Francoise on 20/01/13.
//
//

#ifndef rtml_training_set_h
#define rtml_training_set_h

#include <map>
#include <vector>
#include <set>
#include "gesturesound_phrase.h"
#include "gesture_phrase.h"
#include "notifiable.h"

using namespace std;

#pragma mark -
#pragma mark Base Training Set Class
/*!
 @class _TrainingSetBase
 @brief Base class for the definition of training sets
 @todo class description
 @tparam phraseType Data type of the phrases composing the training set
 @tparam labelType type of the label for each phrase of the training set
 */
template <typename phraseType, typename labelType=int>
class _TrainingSetBase
{
public:
    
    /*!
     @name iterators
     */
    typedef typename  map<int, phraseType*>::iterator phrase_iterator;
    typedef typename  map<int, phraseType*>::const_iterator const_phrase_iterator;
    typedef typename  map<int, labelType>::iterator label_iterator;
    typedef typename  map<int, labelType>::const_iterator const_label_iterator;
    
    map<int, phraseType*> phrases;    //<! Training phrases
    map<int, labelType> phraseLabels; //<! Phrase labels
    set<labelType> allLabels;         //<! set of all existing labels
    
    /*!
     @return iterator to the beginning of phrases
     */
    phrase_iterator begin()
    {
        return phrases.begin();
    }
    
    /*!
     @return iterator to the end of phrases
     */
    phrase_iterator end()
    {
        return phrases.end();
    }
    
    /*!
     @param m index of phrase
     @return iterator to the phrase of index n
     */
    phrase_iterator operator()(int n)
    {
        phrase_iterator pp = phrases.begin();
        for (int i=0; i<n; i++) {
            pp++;
        }
        return pp;
    }
    
#pragma mark -
#pragma mark Constructors
    /*! @name Constructors */
    /*!
     Constructor
     @param _parent parent learning model => the parent is notified each time the training set
     attributes are modified
     */
    _TrainingSetBase(Notifiable* _parent=NULL)
    {
        locked = false;
        defaultLabel = labelType();
        parent = _parent;
        changed = false;
    }
    
    /*!
     Destructor\n
     @warning phrases are only deleted if the training set is unlocked
     @see lock()
     */
    virtual ~_TrainingSetBase()
    {
        if (!locked) {
            for (phrase_iterator it=phrases.begin(); it != phrases.end(); it++)
                delete it->second;
        }
        phrases.clear();
        phraseLabels.clear();
        allLabels.clear();
    }
    
    /*!
     Lock training set to keep the phrases from being deleted at destruction
     */
    void lock()
    {
        locked = true;
    }
    
#pragma mark -
#pragma mark Accessors & tests
    /*! @name accessors and tests */
    /*!
     @return true if the training set is empty (no training phrases)
     */
    bool is_empty() const
    {
        return (phrases.size() == 0);
    }
    
    /*!
     @return size of the training set (number of phrases)
     */
    unsigned int size() const
    {
        return phrases.size();
    }
    
    /*!
     @return true is the training data or attributes have changed
     */
    bool has_changed()
    {
        return changed;
    }
    
    /*!
     set the status of the training set to unchanged
     */
    void set_unchanged()
    {
        changed = false;
    }
    
    /*!
     Set parent model (to be notified when attributes are modified)
     @param _parent parent model
     */
    void set_parent(Notifiable* _parent)
    {
        parent = _parent;
    }
    
    /*!
     checks equality
     @param src training set to compare
     @return true if the training sets are equal (same phrases and labels)
     */
    bool operator==(_TrainingSetBase<phraseType, labelType> const &src)
    {
        if (!this)
            return false;
        
        if (this->defaultLabel != src.defaultLabel)
            return false;
        
        if (this->referencePhrase != src.referencePhrase) // TODO: keep that?
            return false;
        
        for (const_phrase_iterator it=src.phrases.begin(); it != src.phrases.end(); it++)
        {
            if (this->phrases.find(it->first) == this->phrases.end())
                return false;
            
            if (*(this->phrases[it->first]) != *(it->second))
                return false;
        }
        for (const_label_iterator it = src.phraseLabels.begin(); it != src.phraseLabels.end(); it++)
        {
            if (phraseLabels[it->first] != it->second) return false;
        }
        
        return true;
    }
    
    /*!
     checks inequality
     */
    bool operator!=(_TrainingSetBase<phraseType, labelType> const &src)
    {
        return !operator==(src);
    }
    
#pragma mark -
#pragma mark Record training Data
    /*! @name Record training Data */
    /*!
     record training data\n
     A phrase is created if it does not exists at the given index
     @warning this method is only usable if phrases have own data (no shared memory)
     @param phraseIndex index of the phrase
     @param observation observation vector to append to the phrase
     */
    void recordPhrase(int phraseIndex, float *observation)
    {
        if (this->phrases.find(phraseIndex) == this->phrases.end()) {
            resetPhrase(phraseIndex);
            setPhraseLabelToDefault(phraseIndex);
        }
        phrases[phraseIndex]->record(observation);
        changed = true;
    }
    
    /*!
     reset phrase to default\n
     the phrase is created if it does not exists at the given index
     @param phraseIndex index of the phrase
     */
    void resetPhrase(int phraseIndex)
    {
        if (this->phrases.find(phraseIndex) == this->phrases.end())
            phrases[phraseIndex] = new phraseType(referencePhrase);
        else
            *(phrases[phraseIndex]) = referencePhrase;
        changed = true;
    }
    
    /*!
     delete a phrase
     @warning if the training set is locked, the phrases iself is not deleted (only the reference)
     @param phraseIndex index of the phrase
     */
    void deletePhrase(int phraseIndex)
    {
        if (!locked)
            delete phrases[phraseIndex];
        phrases.erase(phraseIndex);
        phraseLabels.erase(phraseIndex);
        changed = true;
    }
    
    /*!
     delete all phrases of a given class
     @warning if the training set is locked, the phrases themselves are not deleted (only their references)
     @param label label of the class to delete
     */
    void deletePhrasesOfClass(labelType label)
    {
        bool contLoop(true);
        while (contLoop) {
            contLoop = false;
            for (label_iterator it=phraseLabels.begin(); it != phraseLabels.end(); it++) {
                if (it->second == label) {
                    deletePhrase(it->first);
                    contLoop = true;
                    break;
                }
            }
        }
        
    }
    
    /*!
     delete all empty phrases
     */
    void deleteEmptyPhrases()
    {
        for (phrase_iterator it=phrases.begin(); it != phrases.end(); it++) {
            if (it->second->is_empty()) {
                deletePhrase(it->first);
            }
        }
        changed = true;
    }
    
    /*!
     delete all phrases
     @warning if the training set is locked, the phrases themselves are not deleted (only their references)
     */
    void clear()
    {
        if (!locked)
            for (phrase_iterator it = this->begin(); it != this->end(); it++) {
                delete it->second;
            }
        phrases.clear();
        phraseLabels.clear();
        changed = true;
    }
    
#pragma mark -
#pragma mark Handle Class Labels
    /*! @name Manipulation of Class Labels */
    /*!
     set default phrase label for new phrases
     */
    void setDefaultLabel(labelType defLabel)
    {
        defaultLabel = defLabel;
    }
    
    /*!
     set label of a phrase to default
     */
    void setPhraseLabelToDefault(int phraseIndex)
    {
        setPhraseLabel(phraseIndex, defaultLabel);
    }
    
    /*!
     set the label of a phrase
     */
    void setPhraseLabel(int phraseIndex, labelType label)
    {
        if (this->phrases.find(phraseIndex) == this->phrases.end())
            throw RTMLException("Training set: phrase does not exist", __FILE__, __FUNCTION__, __LINE__);
        
        phraseLabels[phraseIndex] = label;
        changed = true;
        updateLabelList();
    }
    
    /*!
     create a training set containing all phrases of a given class
     @warning in order to protect the phrases in the current training set, the sub-training set
     returned is locked
     @param label label of the class
     @return a training set containing all the phrases of the given class
     */
    _TrainingSetBase<phraseType, labelType>* getSubTrainingSetForClass(labelType label)
    {
        _TrainingSetBase<phraseType, labelType> *subTS = new _TrainingSetBase();
        subTS->setDefaultLabel(defaultLabel);
        subTS->referencePhrase = referencePhrase;
        
        // Ensure Phrases can't be deleted from a subset
        subTS->lock();
        subTS->changed = true;
        
        for (label_iterator it=phraseLabels.begin(); it != phraseLabels.end(); it++) {
            if (it->second == label) {
                subTS->phrases[it->first] = this->phrases[it->first];
                subTS->phraseLabels[it->first] = label;
            }
        }
        
        return subTS;
    }
    
    /*!
     update the list of all existing labels of the training set
     */
    void updateLabelList()
    {
        allLabels.clear();
        for (label_iterator it=phraseLabels.begin(); it != phraseLabels.end(); it++) {
            allLabels.insert(it->second);
        }
    }
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    /*!
     write training set to stream
     @todo check if complete
     */
    void write(ostream& outStream)
    {
        outStream << "# Training Set\n";
        outStream << "# ===========================\n";
        outStream << "# Number of Phrases\n";
        outStream << phrases.size() << endl;
        outStream << "# Default Label\n";
        outStream << defaultLabel << endl;
        outStream << "# === Reference Phrase\n";
        referencePhrase.write(outStream);
        for (phrase_iterator it = phrases.begin(); it != phrases.end(); it++) {
            outStream << "# === Phrase " << it->first << "\n";
            outStream << "# Index\n";
            outStream << it->first << endl;
            outStream << "# Label\n";
            outStream << phraseLabels[it->first] << endl;
            outStream << "# Content\n";
            it->second->write(outStream);
        }
    }
    
    /*!
     read training set from stream
     @todo check if complete
     */
    void read(istream& inStream)
    {
        skipComments(&inStream);
        
        // Get Number of phrases
        int nbPhrases;
        inStream >> nbPhrases;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        // Get Default Label
        skipComments(&inStream);
        inStream >> defaultLabel;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        // Get reference phrase
        referencePhrase.read(inStream);
        
        // === Get Phrases
        int index;
        labelType label;
        
        for (int p=0; p<nbPhrases; p++) {
            // index
            skipComments(&inStream);
            inStream >> index;
            if (!inStream.good())
                throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
            
            // label
            skipComments(&inStream);
            inStream >> label;
            if (!inStream.good())
                throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
            
            // Content
            phrases[index] = new phraseType(referencePhrase);
            phrases[index]->read(inStream);
            phraseLabels[index] = label;
        }
    }
    
#pragma mark -
#pragma mark Debug
    /*! @name Debug */
    /*!
     Dump training set information to stream
     */
    void dump(ostream& outStream)
    {
        outStream << "# Training Set\n";
        outStream << "# ===========================\n";
        outStream << "# Number of Phrases\n";
        outStream << phrases.size() << endl;
        outStream << "# Default Label\n";
        outStream << defaultLabel << endl;
        for (phrase_iterator it = phrases.begin(); it != phrases.end(); it++) {
            outStream << "# === Phrase " << it->first << ", Label " << phraseLabels[it->first] << endl;
        }
        outStream << endl;
    }
    
#pragma mark -
#pragma mark Python
    /*! @name Python methods */
#ifdef SWIGPYTHON
    /*!
     special python "print" method to get information on the object
     */
    char *__str__() {
        stringstream ss;
        dump(ss);
        string tmp = ss.str();
        char* cstr = strdup(tmp.c_str());
        return cstr;
    }
    
    // TODO: Make class extension in Swig interface file using %extend ?
    /*!
     Append data to a phrase from a numpy array
     @param dimension_total total dimension of the multimodal observation vector
     @param observation multimodal observation vector
     @todo Make class extension in Swig interface file using %extend ?
     */
    void recordPhrase(int phraseIndex, int dimension_total, double *observation)
    {
        float *observation_float = new float[dimension_total];
        for (int d=0; d<dimension_total; d++) {
            observation_float[d] = float(observation[d]);
        }
        
        recordPhrase(phraseIndex, observation_float);
        
        delete[] observation_float;
    }
#endif
    
#pragma mark -
#pragma mark Protected Attributes
    /*! @name Protected Attributes */
protected:
    Notifiable* parent;
    phraseType referencePhrase; //<! Reference phrase: used to store Phrase Attributes
    
    labelType defaultLabel;
    bool changed;
    bool locked;
};



#pragma mark -
#pragma mark Training Set: Class Definition and specializations
/*!
 @class TrainingSet
 @brief Training set
 Adds partial specializations of _TrainingSetBase for specificic types of phrases
 @todo class description
 @tparam phraseType Data type of the phrases composing the training set
 @tparam labelType type of the label for each phrase of the training set
 */
template <typename phraseType, typename labelType>
class TrainingSet : public _TrainingSetBase<phraseType, labelType>
{
public:
    TrainingSet(Notifiable* _parent=NULL) : _TrainingSetBase<phraseType, labelType>(_parent)
    {}
    
    virtual ~TrainingSet() {}
};

#pragma mark Phrase specialization
template <bool ownData, unsigned int nbModalities, typename labelType>
class TrainingSet<Phrase<ownData, nbModalities>, labelType>
: public _TrainingSetBase<Phrase<ownData, nbModalities>, labelType>
{
public:
    typedef typename  map<int, Phrase<ownData, nbModalities>* >::iterator phrase_iterator;
    
    TrainingSet(Notifiable* _parent=NULL)
    : _TrainingSetBase<Phrase<ownData, nbModalities>, labelType>(_parent) {}
    
    virtual ~TrainingSet() {}
    
    /*!
     Connect a phrase to a shared data container
     @param phraseIndex phrase index
     @param _data array of pointers to shared data
     @param _length length of the phrase
     */
    void connect(int phraseIndex, float *_data[nbModalities], int _length)
    {
        if (this->phrases.find(phraseIndex) == this->phrases.end()) {
            this->phrases[phraseIndex] = new Phrase<ownData, nbModalities>(this->referencePhrase);
            this->setPhraseLabelToDefault(phraseIndex);
        }
        this->phrases[phraseIndex]->connect(_data, _length);
        this->changed = true;
    }
    
    /*!
     get dimension of a modality
     @param modality index of the modality
     */
    int get_dimension(int modality=0) const
    {
        return this->referencePhrase.get_dimension(modality);
    }
    
    /*!
     set dimension of a modality
     @param _dimension new dimension
     @param modality index of the modality
     */
    void set_dimension(int _dimension, int modality=0)
    {
        this->referencePhrase.set_dimension(_dimension, modality);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); it++)
            it->second->set_dimension(_dimension, modality);
        if (this->parent)
            this->parent->notify("dimension");
        this->changed = true;
    }
};

#pragma mark GestureSoundPhrase specialization
template <bool ownData, typename labelType>
class TrainingSet< GestureSoundPhrase<ownData>, labelType>
: public _TrainingSetBase< GestureSoundPhrase<ownData>, labelType>
{
public:
    typedef typename  map<int, GestureSoundPhrase<ownData>* >::iterator phrase_iterator;
    
    TrainingSet(Notifiable* _parent=NULL)
    : _TrainingSetBase<GestureSoundPhrase<ownData>, labelType>(_parent) {}
    
    virtual ~TrainingSet() {}
    
    /*!
     Connect a phrase to a shared data container (gesture-sound)
     @param phraseIndex phrase index
     @param _data_gesture pointer to shared gesture data array
     @param _data_sound pointer to shared sound data array
     @param _length length of the phrase
     */
    void connect(int phraseIndex, float *_data_gesture, float *_data_sound, int _length)
    {
        if (this->phrases.find(phraseIndex) == this->phrases.end()) {
            this->phrases[phraseIndex] = new GestureSoundPhrase<ownData>(this->referencePhrase);
            this->setPhraseLabelToDefault(phraseIndex);
        }
        this->phrases[phraseIndex]->connect(_data_gesture, _data_sound, _length);
        this->changed = true;
    }
    
    /*!
     get dimension of the gesture modality
     */
    int get_dimension_gesture() const
    {
        return this->referencePhrase.get_dimension_gesture();
    }
    
    /*!
     set dimension of the gesture modality
     */
    void set_dimension_gesture(int _dimension_gesture)
    {
        this->referencePhrase.set_dimension_gesture(_dimension_gesture);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); it++)
            it->second->set_dimension_gesture(_dimension_gesture);
        if (this->parent)
            this->parent->notify("dimension_gesture");
        this->changed = true;
    }
    
    /*!
     get dimension of the sound modality
     */
    int get_dimension_sound() const
    {
        return this->referencePhrase.get_dimension_sound();
    }
    
    /*!
     set dimension of the sound modality
     */
    void set_dimension_sound(int _dimension_sound)
    {
        this->referencePhrase.set_dimension_sound(_dimension_sound);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); it++)
            it->second->set_dimension_sound(_dimension_sound);
        if (this->parent)
            this->parent->notify("dimension_sound");
        this->changed = true;
    }
};

#pragma mark GesturePhrase specialization
template <bool ownData, typename labelType>
class TrainingSet< GesturePhrase<ownData>, labelType>
: public _TrainingSetBase< GesturePhrase<ownData>, labelType>
{
public:
    typedef typename  map<int, GesturePhrase<ownData>* >::iterator phrase_iterator;
    
    TrainingSet(Notifiable* _parent=NULL)
    : _TrainingSetBase<GesturePhrase<ownData>, labelType>(_parent) {}
    
    virtual ~TrainingSet() {}
    
    /*!
     Connect a phrase to a shared data container (gesture-sound)
     @param phraseIndex phrase index
     @param _data pointer to shared gesture data array
     @param _length length of the phrase
     */
    void connect(int phraseIndex, float *_data, int _length)
    {
        if (this->phrases.find(phraseIndex) == this->phrases.end()) {
            this->phrases[phraseIndex] = new GesturePhrase<ownData>(this->referencePhrase);
            this->setPhraseLabelToDefault(phraseIndex);
        }
        this->phrases[phraseIndex]->connect(_data, _length);
        this->changed = true;
    }
    
    /*!
     get dimension of the gesture modality
     */
    int get_dimension() const
    {
        return this->referencePhrase.get_dimension();
    }
    
    /*!
     set dimension of the gesture modality
     */
    void set_dimension(int _dimension)
    {
        this->referencePhrase.set_dimension(_dimension);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); it++)
            it->second->set_dimension(_dimension);
        if (this->parent)
            this->parent->notify("dimension");
        this->changed = true;
    }
};

#pragma mark -
#pragma mark Python Specializations
/*
 As swig doesn't support partial template specialization with arguments which are
 templates themselves, explicit specialization need to be defined.
 3 templates are specialized here for 3 types of phrases: unimodal, bimodal and gesture-sound.
 */
#ifdef SWIGPYTHON
template <typename labelType>
class TrainingSet<Phrase<true, 1>, labelType>
: public _TrainingSetBase<Phrase<true, 1>, labelType>
{
public:
    typedef  map<int, Phrase<true, 1>* >::iterator phrase_iterator;
    
    TrainingSet(Notifiable* _parent=NULL)
    : _TrainingSetBase<Phrase<true, 1>, labelType>(_parent) {}
    
    virtual ~TrainingSet() {}
    
    int get_dimension(int modality=0) const
    {
        return this->referencePhrase.get_dimension(modality);
    }
    
    void set_dimension(int _dimension, int modality=0)
    {
        this->referencePhrase.set_dimension(_dimension, modality);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); it++)
            it->second->set_dimension(_dimension, modality);
        if (this->parent)
            this->parent->notify("dimension");
        this->changed = true;
    }
};

template <typename labelType>
class TrainingSet<Phrase<true, 2>, labelType>
: public _TrainingSetBase<Phrase<true, 2>, labelType>
{
public:
    typedef  map<int, Phrase<true, 2>* >::iterator phrase_iterator;
    
    TrainingSet(Notifiable* _parent=NULL)
    : _TrainingSetBase<Phrase<true, 2>, labelType>(_parent) {}
    
    virtual ~TrainingSet() {}
    
    int get_dimension(int modality=0) const
    {
        return this->referencePhrase.get_dimension(modality);
    }
    
    void set_dimension(int _dimension, int modality=0)
    {
        this->referencePhrase.set_dimension(_dimension, modality);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); it++)
            it->second->set_dimension(_dimension, modality);
        if (this->parent)
            this->parent->notify("dimension");
        this->changed = true;
    }
};

template <typename labelType>
class TrainingSet< GestureSoundPhrase<true>, labelType>
: public _TrainingSetBase< GestureSoundPhrase<true>, labelType>
{
public:
    typedef map<int, GestureSoundPhrase<true>* >::iterator phrase_iterator;
    
    TrainingSet(Notifiable* _parent=NULL)
    : _TrainingSetBase<GestureSoundPhrase<true>, labelType>(_parent) {}
    
    virtual ~TrainingSet() {}
    
    int get_dimension_gesture() const
    {
        return this->referencePhrase.get_dimension_gesture();
    }
    
    void set_dimension_gesture(int _dimension_gesture)
    {
        this->referencePhrase.set_dimension_gesture(_dimension_gesture);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); it++)
            it->second->set_dimension_gesture(_dimension_gesture);
        if (this->parent)
            this->parent->notify("dimension_gesture");
        this->changed = true;
    }
    
    int get_dimension_sound() const
    {
        return this->referencePhrase.get_dimension_sound();
    }
    
    void set_dimension_sound(int _dimension_sound)
    {
        this->referencePhrase.set_dimension_sound(_dimension_sound);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); it++)
            it->second->set_dimension_sound(_dimension_sound);
        if (this->parent)
            this->parent->notify("dimension_sound");
        this->changed = true;
    }
};

template <typename labelType>
class TrainingSet< GesturePhrase<true>, labelType>
: public _TrainingSetBase< GesturePhrase<true>, labelType>
{
public:
    typedef map<int, GesturePhrase<true>* >::iterator phrase_iterator;
    
    TrainingSet(Notifiable* _parent=NULL)
    : _TrainingSetBase<GesturePhrase<true>, labelType>(_parent) {}
    
    virtual ~TrainingSet() {}
    
    int get_dimension() const
    {
        return this->referencePhrase.get_dimension();
    }
    
    void set_dimension(int _dimension)
    {
        this->referencePhrase.set_dimension(_dimension);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); it++)
            it->second->set_dimension(_dimension);
        if (this->parent)
            this->parent->notify("dimension");
        this->changed = true;
    }
};
#endif

#endif