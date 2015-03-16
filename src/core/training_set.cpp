/*
 * training_set.cpp
 *
 * Multimodal Training Dataset
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
#include "training_set.h"

#pragma mark -
#pragma mark Constructors
TrainingSet::TrainingSet(rtml_flags flags,
                         unsigned int dimension,
                         unsigned int dimension_input)
{
    flags_ = flags;
    locked_ = false;
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
    dst->listeners_ = src.listeners_;
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
    for (set<Listener *>::iterator listen_it = listeners_.begin(); listen_it != listeners_.end(); ++listen_it) {
        (*listen_it)->notify("destruction");
    }
    listeners_.clear();
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

void TrainingSet::add_listener(Listener* listener)
{
    listeners_.insert(listener);
}

void TrainingSet::remove_listener(Listener* listener)
{
    listeners_.erase(listener);
}

unsigned int TrainingSet::dimension()
{
    return dimension_;
}

unsigned int TrainingSet::dimension_input()
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
    
    for (map<Label, TrainingSet>::iterator it = subTrainingSets_.begin() ; it != subTrainingSets_.end() ; ++it)
        it->second.set_dimension(dimension_);
    
    for (set<Listener *>::iterator listen_it = listeners_.begin(); listen_it != listeners_.end(); ++listen_it) {
        (*listen_it)->notify("dimension");
    }
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
    
    for (map<Label, TrainingSet>::iterator it = subTrainingSets_.begin() ; it != subTrainingSets_.end() ; ++it)
        it->second.set_dimension_input(dimension_input_);
    
    for (set<Listener *>::iterator listen_it = listeners_.begin(); listen_it != listeners_.end(); ++listen_it) {
        (*listen_it)->notify("dimension_input");
    }
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
#pragma mark Access Phrases
TrainingSet::phrase_iterator TrainingSet::begin()
{
    return phrases.begin();
}

TrainingSet::phrase_iterator TrainingSet::end()
{
    return phrases.end();
}

TrainingSet::const_phrase_iterator TrainingSet::cbegin() const
{
    return phrases.begin();
}

TrainingSet::const_phrase_iterator TrainingSet::cend() const
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
void TrainingSet::recordPhrase(int phraseIndex, vector<float> const& observation)
{
    if (this->phrases.find(phraseIndex) == this->phrases.end()) {
        resetPhrase(phraseIndex);
        setPhraseLabelToDefault(phraseIndex);
    }
    phrases[phraseIndex]->record(observation);
    has_changed_ = true;
}

void TrainingSet::recordPhrase_input(int phraseIndex, vector<float> const& observation)
{
    if (this->phrases.find(phraseIndex) == this->phrases.end()) {
        resetPhrase(phraseIndex);
        setPhraseLabelToDefault(phraseIndex);
    }
    phrases[phraseIndex]->record_input(observation);
    has_changed_ = true;
}

void TrainingSet::recordPhrase_output(int phraseIndex, vector<float> const& observation)
{
    if (this->phrases.find(phraseIndex) == this->phrases.end()) {
        resetPhrase(phraseIndex);
        setPhraseLabelToDefault(phraseIndex);
    }
    phrases[phraseIndex]->record_output(observation);
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
    if (!locked_) {
        for (phrase_iterator it = this->begin(); it != this->end(); ++it) {
            delete it->second;
            it->second = NULL;
        }
    }
    subTrainingSets_.clear();
    phrases.clear();
    phraseLabels.clear();
    allLabels.clear();
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
        throw out_of_range("Class " + label.as_string() + " does not exist");
    return &(it->second);
}

void TrainingSet::updateSubTrainingSet(Label const& label)
{
    subTrainingSets_[label] = TrainingSet(flags_, dimension_, dimension_input_);
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
    subTrainingSets_.clear();
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
#pragma mark Moments
vector<float> TrainingSet::mean() const
{
    vector<float> mean(dimension_, 0.0);
    unsigned int total_length(0);
    for (const_phrase_iterator it = this->cbegin(); it != this->cend(); ++it)
    {
        for (unsigned int d=0; d<dimension_; d++) {
            for (unsigned int t=0 ; t<it->second->length() ; t++) {
                mean[d] += (*it->second)(t, d);
            }
        }
        total_length += it->second->length();
    }
    
    for (unsigned int d=0; d<dimension_; d++)
        mean[d] /= float(total_length);
    
    return mean;
}

vector<float> TrainingSet::variance() const
{
    vector<float> variance(dimension_);
    vector<float> _mean = mean();
    unsigned int total_length(0);
    for (const_phrase_iterator it = this->cbegin(); it != this->cend(); ++it)
    {
        for (unsigned int d=0; d<dimension_; d++) {
            for (unsigned int t=0 ; t<it->second->length() ; t++) {
                variance[d] += pow((*it->second)(t, d) - _mean[d], 2);
            }
        }
        total_length += it->second->length();
    }
    
    for (unsigned int d=0; d<dimension_; d++)
        variance[d] /= float(total_length);
    
    return variance;
}

#pragma mark -
#pragma mark File IO
JSONNode TrainingSet::to_json() const
{
    JSONNode json_ts(JSON_NODE);
    json_ts.set_name("TrainingSet");
    json_ts.push_back(JSONNode("bimodal", bimodal_));
    json_ts.push_back(JSONNode("dimension", dimension_));
    if (bimodal_)
        json_ts.push_back(JSONNode("dimension_input", dimension_input_));
    json_ts.push_back(JSONNode("size", phrases.size()));
    JSONNode json_deflabel = defaultLabel_.to_json();
    json_deflabel.set_name("defaultlabel");
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
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root.name());
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Number of modalities
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "bimodal")
            throw JSONException("Wrong name: was expecting 'bimodal'", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException("Wrong type: was expecting 'JSON_BOOL'", root_it->name());
        if(bimodal_ != root_it->as_bool()) {
            if (bimodal_)
                throw JSONException("Trying to read an unimodal model in a bimodal_ model.", root.name());
            else
                throw JSONException("Trying to read a bimodal_ model in an unimodal model.", root.name());
        }
        ++root_it;
        
        // Get Dimension
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "dimension")
            throw JSONException("Wrong name: was expecting 'dimension'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        dimension_ = root_it->as_int();
        ++root_it;
        
        // Get Input Dimension if bimodal_
        if (bimodal_){
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->name() != "dimension_input")
                throw JSONException("Wrong name: was expecting 'dimension_input'", root_it->name());
            if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
            dimension_input_ = root_it->as_int();
            ++root_it;
        }
        
        // Get Size: Number of Phrases
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "size")
            throw JSONException("Wrong name: was expecting 'size'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        int ts_size = root_it->as_int();
        ++root_it;
        
        // Get Default label
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "defaultlabel")
            throw JSONException("Wrong name: was expecting 'defaultlabel'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        defaultLabel_.from_json(*root_it);
        ++root_it;
        
        // Get Phrases
        phrases.clear();
        phraseLabels.clear();
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "phrases")
            throw JSONException("Wrong name: was expecting 'phrases'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        for (int i=0 ; i<ts_size ; i++)
        {
            JSONNode::const_iterator array_it = (*root_it)[i].begin();
            // Get Index
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete", array_it->name());
            if (array_it->name() != "index")
                throw JSONException("Wrong name: was expecting 'index'", root_it->name());
            if (array_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
            int phraseIndex = array_it->as_int();
            ++array_it;
            
            // Get Label
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete", array_it->name());
            if (array_it->name() != "label")
                throw JSONException("Wrong name: was expecting 'label'", root_it->name());
            if (array_it->type() != JSON_NODE)
                throw JSONException("Wrong type: was expecting 'JSON_NODE'", root_it->name());
            phraseLabels[phraseIndex].from_json(*array_it);
            updateLabelList();
            ++array_it;
            
            // Get Phrase Content
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete", array_it->name());
            if (array_it->name() != "Phrase")
                throw JSONException("Wrong name: was expecting 'Phrase'", root_it->name());
            if (array_it->type() != JSON_NODE)
                throw JSONException("Wrong type: was expecting 'JSON_NODE'", array_it->name());
            phrases[phraseIndex] = new Phrase(flags_, dimension_, dimension_input_);
            phraseLabels[phraseIndex].from_json(*array_it);
        }
        
        if (ts_size != phrases.size())
            throw JSONException("Number of phrases does not match", root_it->name());
        has_changed_ = true;
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}