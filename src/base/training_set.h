//
// training_set.h
//
// Multimodal training set
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#ifndef rtml_training_set_h
#define rtml_training_set_h

#include "label.h"
#include "phrase.h"
#include "listener.h"
#include <map>
#include <vector>
#include <set>

using namespace std;

#pragma mark -
#pragma mark Base Training Set Class
/*!
 @class _TrainingSetBase
 @brief Base class for the definition of training sets
 @todo class description
 @tparam phraseType Data type of the phrases composing the training set
 */
template <typename phraseType>
class _TrainingSetBase
{
public:
    
    /*!
     @name iterators
     */
    typedef typename  map<int, phraseType*>::iterator phrase_iterator;
    typedef typename  map<int, phraseType*>::const_iterator const_phrase_iterator;
    typedef typename  map<int, Label>::iterator label_iterator;
    typedef typename  map<int, Label>::const_iterator const_label_iterator;
    
    map<int, phraseType*> phrases;    //<! Training phrases
    map<int, Label> phraseLabels; //<! Phrase labels
    set<Label> allLabels;         //<! set of all existing labels
    
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
     @param n index of phrase
     @return iterator to the phrase of index n
     */
    phrase_iterator operator()(int n)
    {
        phrase_iterator pp = phrases.begin();
        for (int i=0; i<n; i++) {
            ++pp;
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
    _TrainingSetBase(Listener* _parent=NULL)
    {
        locked = false;
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
            for (phrase_iterator it=phrases.begin(); it != phrases.end(); ++it)
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
        return phrases.empty();
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
    void set_parent(Listener* _parent)
    {
        parent = _parent;
    }
    
    /*!
     checks equality
     @param src training set to compare
     @return true if the training sets are equal (same phrases and labels)
     */
    bool operator==(_TrainingSetBase<phraseType> const &src)
    {
        if (!this)
            return false;
        
        if (this->defaultLabel != src.defaultLabel)
            return false;
        
        if (this->referencePhrase != src.referencePhrase) // TODO: keep that?
            return false;
        
        for (const_phrase_iterator it=src.phrases.begin(); it != src.phrases.end(); ++it)
        {
            if (this->phrases.find(it->first) == this->phrases.end())
                return false;
            
            if (*(this->phrases[it->first]) != *(it->second))
                return false;
        }
        for (const_label_iterator it = src.phraseLabels.begin(); it != src.phraseLabels.end(); ++it)
        {
            if (phraseLabels[it->first] != it->second) return false;
        }
        
        return true;
    }
    
    /*!
     checks inequality
     */
    bool operator!=(_TrainingSetBase<phraseType> const &src)
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
    void deletePhrasesOfClass(Label label)
    {
        bool contLoop(true);
        while (contLoop) {
            contLoop = false;
            for (label_iterator it=phraseLabels.begin(); it != phraseLabels.end(); ++it) {
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
        for (phrase_iterator it=phrases.begin(); it != phrases.end(); ++it) {
            if (it->second->empty()) {
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
            for (phrase_iterator it = this->begin(); it != this->end(); ++it) {
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
    void setDefaultLabel(Label defLabel)
    {
        defaultLabel = defLabel;
    }
    
    void setDefaultLabel(int intLabel)
    {
        defaultLabel.setInt(intLabel);
    }
    
    void setDefaultLabel(string symLabel)
    {
        defaultLabel.setSym(symLabel);
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
    void setPhraseLabel(int phraseIndex, Label label)
    {
        if (this->phrases.find(phraseIndex) == this->phrases.end())
            throw RTMLException("Training set: phrase does not exist", __FILE__, __FUNCTION__, __LINE__);
        
        phraseLabels[phraseIndex] = label;
        changed = true;
        updateLabelList();
    }
    
    void setPhraseLabel(int phraseIndex, int intLabel)
    {
        Label l;
        l.setInt(intLabel);
        setPhraseLabel(phraseIndex, l);
    }
    
    void setPhraseLabel(int phraseIndex, string symLabel)
    {
        Label l;
        l.setSym(symLabel);
        setPhraseLabel(phraseIndex, l);
    }
    
    Label getPhraseLabel(int phraseIndex)
    {
        return phraseLabels[phraseIndex];
    }
    
    /*!
     create a training set containing all phrases of a given class
     @warning in order to protect the phrases in the current training set, the sub-training set
     returned is locked
     @param label label of the class
     @return a training set containing all the phrases of the given class
     */
    _TrainingSetBase<phraseType>* getSubTrainingSetForClass(Label label)
    {
        _TrainingSetBase<phraseType> *subTS = new _TrainingSetBase();
        subTS->setDefaultLabel(defaultLabel);
        subTS->referencePhrase = referencePhrase;
        
        // Ensure Phrases can't be deleted from a subset
        subTS->lock();
        subTS->changed = true;
        
        int newPhraseIndex(0);
        for (label_iterator it=phraseLabels.begin(); it != phraseLabels.end(); ++it) {
            if (it->second == label) {
                subTS->phrases[newPhraseIndex] = this->phrases[it->first];
                subTS->phraseLabels[newPhraseIndex] = label;
                newPhraseIndex++;
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
        for (label_iterator it=phraseLabels.begin(); it != phraseLabels.end(); ++it) {
            allLabels.insert(it->second);
        }
    }
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    /*!
     Write to JSON Node
     */
    virtual JSONNode to_json() const
    {
        JSONNode json_ts(JSON_NODE);
        json_ts.set_name("Training Set");
        json_ts.push_back(JSONNode("size", phrases.size()));
        JSONNode json_deflabel = defaultLabel.to_json();
        json_deflabel.set_name("default label");
        json_ts.push_back(json_deflabel);
        
        // Add reference phrase
        JSONNode json_refphrase = referencePhrase.to_json();
        json_refphrase.set_name("reference phrase");
        json_ts.push_back(json_refphrase);
        
        // Add phrases
        JSONNode json_phrases(JSON_ARRAY);
        for (const_phrase_iterator it = phrases.begin(); it != phrases.end(); ++it)
        {
            JSONNode json_phrase(JSON_NODE);
            json_phrase.push_back(JSONNode("index", it->first));
            json_phrase.push_back(phraseLabels.at(it->first).to_json());
            json_phrase.push_back(it->second->to_json());
            json_phrases.push_back(json_phrase);
        }
        json_phrases.set_name("phrases");
        json_ts.push_back(json_phrases);
        
        return json_ts;
    }
    
    /*!
     Read from JSON Node
     */
    virtual void from_json(JSONNode root)
    {
        try {
            assert(root.type() == JSON_NODE);
            JSONNode::const_iterator root_it = root.begin();
            
            // Get Size: Number of Phrases
            assert(root_it != root.end());
            assert(root_it->name() == "size");
            assert(root_it->type() == JSON_NUMBER);
            int ts_size = root_it->as_int();
            ++root_it;
            
            // Get Default label
            assert(root_it != root.end());
            assert(root_it->name() == "default label");
            assert(root_it->type() == JSON_NODE);
            defaultLabel.from_json(*root_it);
            ++root_it;
            
            // Get Reference Phrase
            assert(root_it != root.end());
            assert(root_it->name() == "reference phrase");
            assert(root_it->type() == JSON_NODE);
            referencePhrase.from_json(*root_it);
            ++root_it;
            
            // Get Phrases
            phrases.clear();
            phraseLabels.clear();
            assert(root_it != root.end());
            assert(root_it->name() == "phrases");
            assert(root_it->type() == JSON_ARRAY);
            for (int i=0 ; i<ts_size ; i++)
            {
                JSONNode::const_iterator array_it = (*root_it)[i].begin();
                // Get Index
                assert(array_it != root.end());
                assert(array_it->name() == "index");
                assert(array_it->type() == JSON_NUMBER);
                int phraseIndex = array_it->as_int();
                ++array_it;
                
                // Get Label
                assert(array_it != root.end());
                assert(array_it->name() == "label");
                assert(array_it->type() == JSON_NODE);
                phraseLabels[phraseIndex].from_json(*array_it);
                updateLabelList();
                ++array_it;
                
                // Get Phrase Content
                assert(array_it != root.end());
                assert(array_it->name() == "Phrase");
                assert(array_it->type() == JSON_NODE);
                phrases[phraseIndex] = new phraseType(this->referencePhrase);
                phraseLabels[phraseIndex].from_json(*array_it);
            }
            
            assert(ts_size == phrases.size());
            changed = true;
            
        } catch (exception &e) {
            throw RTMLException("Error reading JSON, Node: " + root.name() + " >> " + e.what());
        }
    }
    
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
        if (defaultLabel.type == Label::INT)
            outStream << "INT " << defaultLabel.getInt() << endl;
        else
            outStream << "SYM " << defaultLabel.getSym() << endl;
        outStream << "# === Reference Phrase\n";
        referencePhrase.write(outStream);
        for (phrase_iterator it = phrases.begin(); it != phrases.end(); ++it) {
            outStream << "# === Phrase " << it->first << "\n";
            outStream << "# Index\n";
            outStream << it->first << endl;
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
        
        // Read label
        skipComments(&inStream);
        string lType;
        inStream >> lType;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        if (lType == "INT") {
            int intLab;
            inStream >> intLab;
            if (!inStream.good())
                throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
            this->defaultLabel.setInt(intLab);
        } else {
            string symLab;
            inStream >> symLab;
            if (!inStream.good())
                throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
            this->defaultLabel.setSym(symLab);
        }
        
        // Get reference phrase
        referencePhrase.read(inStream);
        
        // === Get Phrases
        int index;
        for (int p=0; p<nbPhrases; p++) {
            // index
            skipComments(&inStream);
            inStream >> index;
            if (!inStream.good())
                throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
            
            // Content
            phrases[index] = new phraseType(referencePhrase);
            phrases[index]->read(inStream);
            phraseLabels[index] = phrases[index]->label;
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
        if (defaultLabel.type == Label::INT)
            outStream << "INT " << defaultLabel.getInt() << endl;
        else
            outStream << "SYM " << defaultLabel.getSym() << endl;
        for (phrase_iterator it = phrases.begin(); it != phrases.end(); ++it) {
            outStream << "# === Phrase " << it->first << ", Label ";
            if (phraseLabels[it->first].type == Label::INT)
                outStream << "INT " << phraseLabels[it->first].getInt() << endl;
            else
                outStream << "SYM " << phraseLabels[it->first].getSym() << endl;
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
    Listener* parent;
    phraseType referencePhrase; //<! Reference phrase: used to store Phrase Attributes
    
    Label defaultLabel;
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
 */
template <typename phraseType>
class TrainingSet : public _TrainingSetBase<phraseType>
{
public:
    TrainingSet(Listener* _parent=NULL) : _TrainingSetBase<phraseType>(_parent)
    {}
    
    virtual ~TrainingSet() {}
};

#pragma mark Phrase specialization
template <bool ownData, unsigned int nbModalities>
class TrainingSet< Phrase<ownData, nbModalities> >
: public _TrainingSetBase< Phrase<ownData, nbModalities> >
{
public:
    typedef typename  map<int, Phrase<ownData, nbModalities>* >::iterator phrase_iterator;
    
    TrainingSet(Listener* _parent=NULL)
    : _TrainingSetBase< Phrase<ownData, nbModalities> >(_parent) {}
    
    virtual ~TrainingSet() {}
    
    /*!
     Connect a phrase to a shared data container
     @param phraseIndex phrase index
     @param _data array of pointers to shared data
     @param _length length of the phrase
     */
    void connect(int phraseIndex, float *_data[nbModalities], unsigned int _length)
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
    unsigned int get_dimension(unsigned int modality=0) const
    {
        return this->referencePhrase.get_dimension(modality);
    }
    
    /*!
     set dimension of a modality
     @param _dimension new dimension
     @param modality index of the modality
     */
    void set_dimension(unsigned int _dimension, unsigned int modality=0)
    {
        this->referencePhrase.set_dimension(_dimension, modality);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); ++it)
            it->second->set_dimension(_dimension, modality);
        if (this->parent)
            this->parent->notify("dimension");
        this->changed = true;
    }
};

#pragma mark GestureSoundPhrase specialization
template <bool ownData>
class TrainingSet< GestureSoundPhrase<ownData> >
: public _TrainingSetBase< GestureSoundPhrase<ownData> >
{
public:
    typedef typename  map<int, GestureSoundPhrase<ownData>* >::iterator phrase_iterator;
    
    TrainingSet(Listener* _parent=NULL)
    : _TrainingSetBase< GestureSoundPhrase<ownData> >(_parent) {}
    
    virtual ~TrainingSet() {}
    
    /*!
     Connect a phrase to a shared data container (gesture-sound)
     @param phraseIndex phrase index
     @param _data_gesture pointer to shared gesture data array
     @param _data_sound pointer to shared sound data array
     @param _length length of the phrase
     */
    void connect(int phraseIndex, float *_data_gesture, float *_data_sound, unsigned int _length)
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
    unsigned int get_dimension_gesture() const
    {
        return this->referencePhrase.get_dimension_gesture();
    }
    
    /*!
     set dimension of the gesture modality
     */
    void set_dimension_gesture(unsigned int _dimension_gesture)
    {
        this->referencePhrase.set_dimension_gesture(_dimension_gesture);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); ++it)
            it->second->set_dimension_gesture(_dimension_gesture);
        if (this->parent)
            this->parent->notify("dimension_gesture");
        this->changed = true;
    }
    
    /*!
     get dimension of the sound modality
     */
    unsigned int get_dimension_sound() const
    {
        return this->referencePhrase.get_dimension_sound();
    }
    
    /*!
     set dimension of the sound modality
     */
    void set_dimension_sound(unsigned int _dimension_sound)
    {
        this->referencePhrase.set_dimension_sound(_dimension_sound);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); ++it)
            it->second->set_dimension_sound(_dimension_sound);
        if (this->parent)
            this->parent->notify("dimension_sound");
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
template<>
class TrainingSet< Phrase<true, 1> >
: public _TrainingSetBase< Phrase<true, 1> >
{
public:
    typedef  map<int, Phrase<true, 1>* >::iterator phrase_iterator;
    
    TrainingSet(Listener* _parent=NULL)
    : _TrainingSetBase< Phrase<true, 1> >(_parent) {}
    
    virtual ~TrainingSet() {}
    
    int get_dimension(int modality=0) const
    {
        return this->referencePhrase.get_dimension(modality);
    }
    
    void set_dimension(int _dimension, int modality=0)
    {
        this->referencePhrase.set_dimension(_dimension, modality);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); ++it)
            it->second->set_dimension(_dimension, modality);
        if (this->parent)
            this->parent->notify("dimension");
        this->changed = true;
    }
};

template<>
class TrainingSet< Phrase<true, 2> >
: public _TrainingSetBase< Phrase<true, 2> >
{
public:
    typedef  map<int, Phrase<true, 2>* >::iterator phrase_iterator;
    
    TrainingSet(Listener* _parent=NULL)
    : _TrainingSetBase< Phrase<true, 2> >(_parent) {}
    
    virtual ~TrainingSet() {}
    
    int get_dimension(int modality=0) const
    {
        return this->referencePhrase.get_dimension(modality);
    }
    
    void set_dimension(int _dimension, int modality=0)
    {
        this->referencePhrase.set_dimension(_dimension, modality);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); ++it)
            it->second->set_dimension(_dimension, modality);
        if (this->parent)
            this->parent->notify("dimension");
        this->changed = true;
    }
};

template<>
class TrainingSet< GestureSoundPhrase<true> >
: public _TrainingSetBase< GestureSoundPhrase<true> >
{
public:
    typedef map<int, GestureSoundPhrase<true>* >::iterator phrase_iterator;
    
    TrainingSet(Listener* _parent=NULL)
    : _TrainingSetBase< GestureSoundPhrase<true> >(_parent) {}
    
    virtual ~TrainingSet() {}
    
    int get_dimension_gesture() const
    {
        return this->referencePhrase.get_dimension_gesture();
    }
    
    void set_dimension_gesture(int _dimension_gesture)
    {
        this->referencePhrase.set_dimension_gesture(_dimension_gesture);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); ++it)
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
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); ++it)
            it->second->set_dimension_sound(_dimension_sound);
        if (this->parent)
            this->parent->notify("dimension_sound");
        this->changed = true;
    }
};

template<>
class TrainingSet< GesturePhrase<true> >
: public _TrainingSetBase< GesturePhrase<true> >
{
public:
    typedef map<int, GesturePhrase<true>* >::iterator phrase_iterator;
    
    TrainingSet(Listener* _parent=NULL)
    : _TrainingSetBase< GesturePhrase<true> >(_parent) {}
    
    virtual ~TrainingSet() {}
    
    int get_dimension() const
    {
        return this->referencePhrase.get_dimension();
    }
    
    void set_dimension(int _dimension)
    {
        this->referencePhrase.set_dimension(_dimension);
        for (phrase_iterator it=this->phrases.begin(); it != this->phrases.end(); ++it)
            it->second->set_dimension(_dimension);
        if (this->parent)
            this->parent->notify("dimension");
        this->changed = true;
    }
};
#endif

#endif