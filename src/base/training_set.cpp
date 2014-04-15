//
// training_set.cpp
//
// Multimodal training set
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#include "training_set.h"

TrainingSet::phrase_iterator TrainingSet::begin()
{
    return phrases.begin();
}

TrainingSet::phrase_iterator TrainingSet::end()
{
    return phrases.end();
}

TrainingSet::phrase_iterator TrainingSet::operator()(int n)
{
    phrase_iterator pp = phrases.begin();
    for (int i=0; i<n; i++) {
        ++pp;
    }
    return pp;
}

#pragma mark -
#pragma mark Constructors
TrainingSet::TrainingSet(rtml_flags flags,
                         Listener* parent,
                         unsigned int dimension,
                         unsigned int dimension_input)
{
    flags_ = flags;
    locked_ = false;
    parent_ = parent;
    has_changed_ = false;
    
    owns_data = !(flags_ & SHARED_MEMORY);
    dimension_ = dimension;
    bimodal_ = (flags_ & BIMODAL);
    dimension_input_ = bimodal_ ? dimension_input : 0;
}

TrainingSet::TrainingSet(TrainingSet const& src)
{
    _copy(this, src);
}

TrainingSet& TrainingSet::operator=(TrainingSet const& src)
{
    if(this != &src)
        _copy(this, src);
    return *this;
}

void TrainingSet::_copy(TrainingSet *dst, TrainingSet const& src)
{
    dst->clear();
    dst->flags_ = src.flags_;
    dst->owns_data = src.owns_data;
    dst->bimodal_ = src.bimodal_;
    dst->dimension_ = src.dimension_;
    dst->dimension_input_ = src.dimension_input_;
    dst->parent_ = src.parent_;
    dst->defaultLabel_ = src.defaultLabel_;
    dst->has_changed_ = true;
    dst->locked_ = src.locked_;
    dst->phrases = src.phrases;
    dst->phraseLabels = src.phraseLabels;
    dst->allLabels = src.allLabels;
    dst->subTrainingSets_ = src.subTrainingSets_;
}

TrainingSet::~TrainingSet()
{
    if (!locked_) {
        for (phrase_iterator it=phrases.begin(); it != phrases.end(); ++it)
            delete it->second;
    }
    subTrainingSets_.clear();
    phrases.clear();
    phraseLabels.clear();
    allLabels.clear();
}

void TrainingSet::lock()
{
    locked_ = true;
}

#pragma mark -
#pragma mark Accessors & tests
bool TrainingSet::is_bimodal() const
{
    return bimodal_;
}

bool TrainingSet::is_empty() const
{
    return phrases.empty();
}

unsigned int TrainingSet::size() const
{
    return phrases.size();
}

bool TrainingSet::has_changed()
{
    return has_changed_;
}

void TrainingSet::set_unchanged()
{
    has_changed_ = false;
}

void TrainingSet::set_parent(Listener* parent)
{
    parent_ = parent;
}

unsigned int TrainingSet::get_dimension()
{
    return dimension_;
}

unsigned int TrainingSet::get_dimension_input()
{
    return dimension_input_;
}

void TrainingSet::set_dimension(unsigned int dimension)
{
    if (dimension == dimension_)
        return;
    
    if (dimension < 1)
        throw domain_error("the dimension of a phrase must be striclty positive");
    
    dimension_ = dimension;
    for (phrase_iterator p = phrases.begin(); p != phrases.end(); p++) {
        p->second->set_dimension(dimension_);
    }
    
    if (this->parent_)
        this->parent_->notify("dimension");
}

void TrainingSet::set_dimension_input(unsigned int dimension_input)
{
    if (!bimodal_)
        throw runtime_error("the training set is not bimodal_");
    
    if (dimension_input == dimension_input_)
        return;
    
    if (dimension_input >= dimension_)
        throw invalid_argument("The dimension of the input modality must not exceed the total dimension.");
    
    dimension_input_ = dimension_input;
    for (phrase_iterator p = phrases.begin(); p != phrases.end(); p++) {
        p->second->set_dimension_input(dimension_input_);
    }
    
    if (this->parent_)
        this->parent_->notify("dimension_input");
}

bool TrainingSet::operator==(TrainingSet const &src)
{
    if (!this)
        return false;
    
    if (this->owns_data != src.owns_data)
        return false;
    
    if (this->bimodal_ != src.bimodal_)
        return false;
    
    if (this->dimension_ != src.dimension_)
        return false;
    
    if (this->dimension_input_ != src.dimension_input_)
        return false;
    
    if (this->defaultLabel_ != src.defaultLabel_)
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

bool TrainingSet::operator!=(TrainingSet const &src)
{
    return !operator==(src);
}

#pragma mark -
#pragma mark Connect Phrases
void TrainingSet::connect(int phraseIndex, float *pointer_to_data, unsigned int length)
{
    if (this->phrases.find(phraseIndex) == this->phrases.end()) {
        resetPhrase(phraseIndex);
        setPhraseLabelToDefault(phraseIndex);
    }
    phrases[phraseIndex]->connect(pointer_to_data, length);
    has_changed_ = true;
}

void TrainingSet::connect(int phraseIndex,
                          float *pointer_to_data_input,
                          float *pointer_to_data_output,
                          unsigned int length)
{
    if (this->phrases.find(phraseIndex) == this->phrases.end()) {
        resetPhrase(phraseIndex);
        setPhraseLabelToDefault(phraseIndex);
    }
    phrases[phraseIndex]->connect(pointer_to_data_input, pointer_to_data_output, length);
    has_changed_ = true;
}

#pragma mark -
#pragma mark Record training Data
void TrainingSet::recordPhrase(int phraseIndex, float *observation)
{
    if (this->phrases.find(phraseIndex) == this->phrases.end()) {
        resetPhrase(phraseIndex);
        setPhraseLabelToDefault(phraseIndex);
    }
    phrases[phraseIndex]->record(observation);
    has_changed_ = true;
}

void TrainingSet::resetPhrase(int phraseIndex)
{
    if (this->phrases.find(phraseIndex) != this->phrases.end())
        delete phrases[phraseIndex];
    phrases[phraseIndex] = new Phrase(flags_, dimension_, dimension_input_);
    has_changed_ = true;
}

void TrainingSet::deletePhrase(int phraseIndex)
{
    if (!locked_)
        delete phrases[phraseIndex];
    phrases.erase(phraseIndex);
    phraseLabels.erase(phraseIndex);
    updateLabelList();
    has_changed_ = true;
}

void TrainingSet::deletePhrasesOfClass(Label const& label)
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

void TrainingSet::deleteEmptyPhrases()
{
    for (phrase_iterator it=phrases.begin(); it != phrases.end(); ++it) {
        if (it->second->is_empty()) {
            deletePhrase(it->first);
        }
    }
}

void TrainingSet::clear()
{
    if (!locked_)
        for (phrase_iterator it = this->begin(); it != this->end(); ++it) {
            delete it->second;
            it->second = NULL;
        }
    subTrainingSets_.clear();
    phrases.clear();
    phraseLabels.clear();
    has_changed_ = true;
}

#pragma mark -
#pragma mark Handle Class Labels
void TrainingSet::setDefaultLabel(Label const& defLabel)
{
    defaultLabel_ = defLabel;
}

void TrainingSet::setPhraseLabelToDefault(int phraseIndex)
{
    setPhraseLabel(phraseIndex, defaultLabel_);
}

void TrainingSet::setPhraseLabel(int phraseIndex, Label const& label)
{
    if (this->phrases.find(phraseIndex) == this->phrases.end())
        throw out_of_range("Training set: phrase does not exist");
    
    phraseLabels[phraseIndex] = label;
    updateLabelList();
    has_changed_ = true;
}

Label TrainingSet::getPhraseLabel(int phraseIndex)
{
    return phraseLabels[phraseIndex];
}

TrainingSet* TrainingSet::getSubTrainingSetForClass(Label const& label)
{
    updateSubTrainingSet(label);
    map<Label, TrainingSet>::iterator it = subTrainingSets_.find(label);
    if (it == subTrainingSets_.end())
        throw out_of_range("Label " + label.as_string() + " Does not exist");
    return &(it->second);
}

void TrainingSet::updateSubTrainingSet(Label const& label)
{
    TrainingSet defts(flags_, NULL, dimension_, dimension_input_);
    subTrainingSets_[label] = TrainingSet(flags_, NULL, dimension_, dimension_input_);
    subTrainingSets_[label].setDefaultLabel(label);
    subTrainingSets_[label].lock();
    int newPhraseIndex(0);
    for (label_iterator it=phraseLabels.begin(); it != phraseLabels.end(); ++it) {
        if (it->second == label) {
            subTrainingSets_[label].phrases[newPhraseIndex] = this->phrases[it->first];
            subTrainingSets_[label].setPhraseLabel(newPhraseIndex, label);
            newPhraseIndex++;
        }
    }
}

void TrainingSet::updateSubTrainingSets()
{
    for (set<Label>::iterator label_it = allLabels.begin(); label_it != allLabels.end(); ++label_it) {
        updateSubTrainingSet(*label_it);
    }
}

void TrainingSet::updateLabelList()
{
    allLabels.clear();
    for (label_iterator it=phraseLabels.begin(); it != phraseLabels.end(); ++it) {
        allLabels.insert(it->second);
    }
}

#pragma mark -
#pragma mark File IO
JSONNode TrainingSet::to_json() const
{
    JSONNode json_ts(JSON_NODE);
    json_ts.set_name("Training Set");
    json_ts.push_back(JSONNode("bimodal_", bimodal_));
    json_ts.push_back(JSONNode("dimension", dimension_));
    if (bimodal_)
        json_ts.push_back(JSONNode("dimension_input_", dimension_input_));
    json_ts.push_back(JSONNode("size", phrases.size()));
    JSONNode json_deflabel = defaultLabel_.to_json();
    json_deflabel.set_name("default label");
    json_ts.push_back(json_deflabel);
    
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

void TrainingSet::from_json(JSONNode root)
{
    if (!owns_data)
        throw runtime_error("Cannot read Training Set with Shared memory");
    
    try {
        assert(root.type() == JSON_NODE);
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Number of modalities
        assert(root_it != root.end());
        assert(root_it->name() == "bimodal_");
        assert(root_it->type() == JSON_BOOL);
        if(bimodal_ != root_it->as_bool()) {
            if (bimodal_)
                throw JSONException("Trying to read an unimodal model in a bimodal_ model.", root.name());
            else
                throw JSONException("Trying to read a bimodal_ model in an unimodal model.", root.name());
        }
        ++root_it;
        
        // Get Dimension
        assert(root_it != root.end());
        assert(root_it->name() == "dimension");
        assert(root_it->type() == JSON_NUMBER);
        dimension_ = root_it->as_int();
        ++root_it;
        
        // Get Input Dimension if bimodal_
        if (bimodal_){
            assert(root_it != root.end());
            assert(root_it->name() == "dimension_input_");
            assert(root_it->type() == JSON_NUMBER);
            dimension_input_ = root_it->as_int();
            ++root_it;
        }
        
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
        defaultLabel_.from_json(*root_it);
        ++root_it;
        
        // TODO: write flags_, dimensions...
        
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
            phrases[phraseIndex] = new Phrase(flags_, dimension_, dimension_input_);
            phraseLabels[phraseIndex].from_json(*array_it);
        }
        
        assert(ts_size == phrases.size());
        has_changed_ = true;
        
    } catch (JSONException &e) {
        throw JSONException(e);
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}

#pragma mark -
#pragma mark Debug
void TrainingSet::dump(ostream& outStream)
{
    outStream << "# Training Set\n";
    outStream << "# ===========================\n";
    outStream << "# Number of Phrases\n";
    outStream << phrases.size() << endl;
    outStream << "# Default Label\n";
    if (defaultLabel_.type == Label::INT)
        outStream << "INT " << defaultLabel_.getInt() << endl;
    else
        outStream << "SYM " << defaultLabel_.getSym() << endl;
    for (phrase_iterator it = phrases.begin(); it != phrases.end(); ++it) {
        outStream << "# === Phrase " << it->first << ", Label ";
        if (phraseLabels[it->first].type == Label::INT)
            outStream << "INT " << phraseLabels[it->first].getInt() << endl;
        else
            outStream << "SYM " << phraseLabels[it->first].getSym() << endl;
    }
    outStream << endl;
}