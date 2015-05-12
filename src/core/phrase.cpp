/*
 * phrase.cpp
 *
 * Multimodal data phrase
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

#include "phrase.h"

#pragma mark Constructors
xmm::Phrase::Phrase(xmm_flags flags,
               unsigned int dimension,
               unsigned int dimension_input)
: owns_data_(!(flags & SHARED_MEMORY)),
  bimodal_(flags & BIMODAL),
  is_empty_(true),
  length_(0),
  length_input_(0),
  length_output_(0),
  max_length_(0),
  dimension_(dimension),
  dimension_input_(bimodal_ ? dimension_input : 0)
{
    data = new float*[bimodal_ ? 2 : 1];
    data[0] = NULL;
    if (bimodal_)
        data[1] = NULL;
}

xmm::Phrase::Phrase(Phrase const& src)
{
    _copy(this, src);
}

xmm::Phrase& xmm::Phrase::operator=(Phrase const& src)
{
    if(this != &src)
        _copy(this, src);
    return *this;
}

void xmm::Phrase::_copy(Phrase *dst, Phrase const& src)
{
    if (owns_data_) {
        
        if (dst->data) {
            if (bimodal_)
                try {
                    delete[] dst->data[1];
                } catch (std::exception& e) {}
            try {
                delete[] dst->data[0];
            } catch (std::exception& e) {}
        }
        try {
            delete[] dst->data;
        } catch (std::exception& e) {}
        dst->data = NULL;
    }
    dst->owns_data_ = src.owns_data_;
    dst->bimodal_ = src.bimodal_;
    dst->is_empty_ = src.is_empty_;
    dst->dimension_ = src.dimension_;
    dst->dimension_input_ = src.dimension_input_;
    dst->max_length_ = src.max_length_;
    dst->length_ = src.length_;
    dst->length_input_ = src.length_input_;
    dst->length_output_ = src.length_output_;
    dst->column_names_ = src.column_names_;
    
    if (owns_data_)
    {
        dst->data = new float*[dst->bimodal_ ? 2 : 1];
        if (dst->max_length_ > 0) {
            unsigned int modality_dim = dst->bimodal_ ? dst->dimension_input_ : dst->dimension_;
            dst->data[0] = new float[dst->max_length_ * modality_dim];
            std::copy(src.data[0], src.data[0] + dst->max_length_ * modality_dim, dst->data[0]);
            if (bimodal_) {
                modality_dim = dst->dimension_ - dst->dimension_input_;
                dst->data[1] = new float[dst->max_length_ * modality_dim];
                std::copy(src.data[1], src.data[1] + dst->max_length_ * modality_dim, dst->data[1]);
            }
        }
    }
    else
    {
        dst->data[0] = src.data[0];
        if (bimodal_)
            dst->data[1] = src.data[1];
    }
}

xmm::Phrase::~Phrase()
{
    if (owns_data_) {
        if (bimodal_) {
            delete[] data[1];
            data[1] = NULL;
        }
        delete[] data[0];
        data[0] = NULL;
    }
    delete[] data;
    data = NULL;
    length_ = 0;
    length_input_ = 0;
    length_output_ = 0;
    max_length_ = 0;
    is_empty_ = true;
}

#pragma mark Tests
bool xmm::Phrase::is_empty() const
{
    return is_empty_;
}

bool xmm::Phrase::operator==(Phrase const& src)
{
    if (this->length_ != src.length_) return false;
    if (this->dimension_ != src.dimension_) return false;
    if (!this->data || !src.data)
        return false;
    if (!this->data[0] || !*src.data[0])
        return false;
    if (this->data[0] != src.data[0])
        return false;
    if (this->bimodal_) {
        if (this->dimension_input_ != src.dimension_input_) return false;
        if (this->length_input_ != src.length_input_) return false;
        if (this->length_output_ != src.length_output_) return false;
        if (this->data[1] != src.data[1])
            return false;
    }
    return true;
}

bool xmm::Phrase::operator!=(Phrase const& src)
{
    return !(operator==(src));
}

#pragma mark Accessors
unsigned int xmm::Phrase::length() const
{
    return length_;
}

void xmm::Phrase::trim(unsigned int phraseLength)
{
    if (length_ > phraseLength) {
        length_ = phraseLength;
        length_input_ = phraseLength;
        length_output_ = phraseLength;
    }
}

void xmm::Phrase::trim()
{
    if (bimodal_)
        length_ = (length_output_ > length_input_) ? length_input_ : length_output_;
}

unsigned int xmm::Phrase::dimension() const
{
    return dimension_;
}

unsigned int xmm::Phrase::dimension_input() const
{
    if (!bimodal_)
        throw std::runtime_error("Phrase is not Bimodal");
    return dimension_input_;
}

unsigned int xmm::Phrase::dimension_output() const
{
    if (!bimodal_)
        throw std::runtime_error("Phrase is not Bimodal");
    return dimension_ - dimension_input_;
}

void xmm::Phrase::set_dimension(unsigned int dimension)
{
    if (dimension == dimension_)
        return;
    
    if (dimension < 1)
        throw std::domain_error("the dimension of a phrase must be striclty positive");
    
    unsigned int modality(0);
    unsigned int modalitydim_src(dimension_);
    unsigned int modalitydim_dst(dimension);
    if (bimodal_) {
        if (dimension < 2)
            throw std::domain_error("the dimension of bimodal_ phrase must be > 2");
        modality = 1;
        modalitydim_src -= this->dimension_input_;
        modalitydim_dst -= this->dimension_input_;
    }
    
    if (owns_data_) {
        data[modality] = reallocate(data[modality],
                                    max_length_ * modalitydim_src,
                                    max_length_ * modalitydim_dst);
    }
    column_names_.resize(dimension);
    this->dimension_ = dimension;
}

void xmm::Phrase::set_dimension_input(unsigned int dimension_input)
{
    if (!bimodal_)
        throw std::runtime_error("the phrase is not bimodal_");
    
    if (dimension_input == dimension_input_)
        return;
    
    if (dimension_input >= dimension_)
        throw std::invalid_argument("The dimension of the input modality must not exceed the total dimension.");
    
    if (owns_data_) {
        data[0] = reallocate(data[0],
                             max_length_ * dimension_input_,
                             max_length_ * dimension_input);
    }
    dimension_input_ = dimension_input;
}

#pragma mark Connect (shared data)
void xmm::Phrase::connect(float *pointer_to_data,
             unsigned int length)
{
    if (owns_data_) throw std::runtime_error("Cannot connect a phrase with own data");
    if (bimodal_) throw std::runtime_error("Cannot connect a single array, use 'connect_input' and 'connect_output'");
    
    data[0] = pointer_to_data;
    length_ = length;
    is_empty_ = false;
}

void xmm::Phrase::connect(float *pointer_to_data_input, float *pointer_to_data_output, unsigned int length)
{
    if (owns_data_) throw std::runtime_error("Cannot connect a phrase with own data");
    if (!bimodal_) throw std::runtime_error("This phrase is unimodal, use 'connect'");
    
    data[0] = pointer_to_data_input;
    data[1] = pointer_to_data_output;
    length_input_ = length;
    length_output_ = length;
    trim();
    is_empty_ = false;
}

void xmm::Phrase::connect_input(float *pointer_to_data,
                   unsigned int length)
{
    if (owns_data_) throw std::runtime_error("Cannot connect a phrase with own data");
    if (!bimodal_) throw std::runtime_error("This phrase is unimodal, use 'connect'");
    
    data[0] = pointer_to_data;
    length_input_ = length;
    trim();
    is_empty_ = false;
}

void xmm::Phrase::connect_output(float *pointer_to_data,
                    unsigned int length)
{
    if (owns_data_) throw std::runtime_error("Cannot connect a phrase with own data");
    if (!bimodal_) throw std::runtime_error("This phrase is unimodal, use 'connect'");
    
    data[1] = pointer_to_data;
    length_output_ = length;
    trim();
    is_empty_ = false;
}

void xmm::Phrase::disconnect()
{
    if (owns_data_) throw std::runtime_error("Cannot disconnect a phrase with own data");
    data[0] = NULL;
    if (bimodal_)
        data[1] = NULL;
    length_ = 0;
    length_input_ = 0;
    length_output_ = 0;
    is_empty_ = true;
}

#pragma mark Record (Own Data)
void xmm::Phrase::record(std::vector<float> const& observation)
{
    if (!owns_data_) throw std::runtime_error("Cannot record in shared data phrase");
    if (bimodal_ && length_input_ != length_output_)
        throw std::runtime_error("Cannot record bimodal_ phrase in synchronous mode: modalities have different length");
    if (observation.size() != dimension_)
        throw std::invalid_argument("Observation has wrong dimension");
    
    if (length_ >= max_length_ || max_length_ == 0) {
        reallocate_length();
    }
    
    if (bimodal_) {
        copy(observation.begin(),
             observation.begin() + dimension_input_,
             data[0] + length_input_ * dimension_input_);
        copy(observation.begin() + dimension_input_,
             observation.begin() + dimension_,
             data[1] + length_output_ * (dimension_ - dimension_input_));
        length_input_++;
        length_output_++;
    } else {
        copy(observation.begin(),
             observation.end(),
             data[0] + length_ * dimension_);
    }
    
    length_++;
    is_empty_ = false;
}

void xmm::Phrase::record_input(std::vector<float> const& observation)
{
    if (!owns_data_) throw std::runtime_error("Cannot record in shared data phrase");
    if (!bimodal_) throw std::runtime_error("this phrase is unimodal, use 'record'");
    if (observation.size() != dimension_input_)
        throw std::invalid_argument("Observation has wrong dimension");

    if (length_input_ >= max_length_ || max_length_ == 0) {
        reallocate_length();
    }
    
    copy(observation.begin(),
         observation.end(),
         data[0] + length_input_ * dimension_input_);
    length_input_++;
    trim();
    is_empty_ = false;
}

void xmm::Phrase::record_output(std::vector<float> const& observation)
{
    if (!owns_data_) throw std::runtime_error("Cannot record in shared data phrase");
    if (!bimodal_) throw std::runtime_error("this phrase is unimodal, use 'record'");
    
    if (observation.size() != dimension_ - dimension_input_)
        throw std::invalid_argument("Observation has wrong dimension");

    if (length_output_ >= max_length_ || max_length_ == 0) {
        reallocate_length();
    }
    
    copy(observation.begin(),
         observation.end(),
         data[1] + length_output_ * (dimension_ - dimension_input_));
    length_output_++;
    trim();
    is_empty_ = false;
}

void xmm::Phrase::reallocate_length()
{
    unsigned int modality_dim = bimodal_ ? dimension_input_ : dimension_;
    data[0] = reallocate<float>(data[0],
                                max_length_ * modality_dim,
                                (max_length_ + ALLOC_BLOCKSIZE) * modality_dim);
    if (bimodal_) {
        modality_dim = dimension_ - dimension_input_;
        data[1] = reallocate<float>(data[1],
                                    max_length_ * modality_dim,
                                    (max_length_ + ALLOC_BLOCKSIZE) * modality_dim);
    }
    max_length_ += ALLOC_BLOCKSIZE;
}

void xmm::Phrase::clear()
{
    if (!owns_data_) throw std::runtime_error("Cannot clear a shared data phrase");
    
    length_ = 0;
    length_input_ = 0;
    length_output_ = 0;
    is_empty_ = true;
}

#pragma mark Access Data
float xmm::Phrase::at(unsigned int index, unsigned int dim) const
{
    if (index >= length_)
        throw std::out_of_range("Phrase: index out of bounds");
    if (dim >= dimension_)
        throw std::out_of_range("Phrase: dimension out of bounds");
    if (bimodal_) {
        if (dim < dimension_input_)
            return data[0][index * dimension_input_ + dim];
        return data[1][index * (dimension_ - dimension_input_) + dim - dimension_input_];
    } else {
        return data[0][index * dimension_ + dim];
    }
}

float xmm::Phrase::operator()(unsigned int index, unsigned int dim) const
{
    return at(index, dim);
}

float* xmm::Phrase::get_dataPointer(unsigned int index) const
{
    if (index >= length_) throw std::out_of_range("Phrase: index out of bounds");
    if (bimodal_) throw std::runtime_error("this phrase is bimodal_, use 'get_dataPointer_input' and 'get_dataPointer_output'");
    return data[0] + index * dimension_;
}

float* xmm::Phrase::get_dataPointer_input(unsigned int index) const
{
    if (index >= length_) throw std::out_of_range("Phrase: index out of bounds");
    if (!bimodal_) throw std::runtime_error("this phrase is unimodal, use 'get_dataPointer'");
    return data[0] + index * dimension_input_;
}

float* xmm::Phrase::get_dataPointer_output(unsigned int index) const
{
    if (index >= length_) throw std::out_of_range("Phrase: index out of bounds");
    if (!bimodal_) throw std::runtime_error("this phrase is unimodal, use 'get_dataPointer'");
    return data[1] + index * (dimension_ - dimension_input_);
}

#pragma mark JSON I/O
JSONNode xmm::Phrase::to_json() const
{
    JSONNode json_phrase(JSON_NODE);
    json_phrase.set_name("Phrase");
    
    json_phrase.push_back(JSONNode("bimodal", bimodal_));
    json_phrase.push_back(JSONNode("dimension", dimension_));
    json_phrase.push_back(JSONNode("length", length_));
    json_phrase.push_back(vector2json(column_names_, "column_names"));
    if (bimodal_) {
        json_phrase.push_back(JSONNode("dimension_input", dimension_input_));
        json_phrase.push_back(array2json(data[0], length_ * dimension_input_, "data_input"));
        json_phrase.push_back(array2json(data[1], length_ * (dimension_ - dimension_input_), "data_output"));
    } else {
        json_phrase.push_back(array2json(data[0], length_ * dimension_, "data"));
    }
    
    return json_phrase;
}

void xmm::Phrase::from_json(JSONNode root)
{
    if (!owns_data_)
        throw std::runtime_error("Cannot read Phrase with Shared memory");
    
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root.name());
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Number of modalities
        root_it = root.find("bimodal");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException("Wrong type for node 'bimodal': was expecting 'JSON_BOOL'", root_it->name());
        if(bimodal_ != root_it->as_bool()) {
            if (bimodal_)
                throw JSONException("Trying to read an unimodal model in a bimodal model.", root.name());
            else
                throw JSONException("Trying to read a bimodal model in an unimodal model.", root.name());
        }
        
        // Get Dimension
        root_it = root.find("dimension");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'dimension': was expecting 'JSON_NUMBER'", root_it->name());
        dimension_ = static_cast<unsigned int>(root_it->as_int());
        
        // Get Length
        root_it = root.find("length");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'length': was expecting 'JSON_NUMBER'", root_it->name());
        length_ = static_cast<unsigned int>(root_it->as_int());
        length_input_ = length_;
        length_output_ = length_;
        
        // Get Column Names
        column_names_.assign(dimension_, "");
        root_it = root.find("column_names");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_ARRAY)
                throw JSONException("Wrong type for node 'column_names': was expecting 'JSON_ARRAY'", root_it->name());
            json2vector(*root_it, column_names_, dimension_);
        }
        
        // Get Input Dimension if bimodal_
        if (bimodal_){
            root_it = root.find("dimension_input");
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type for node 'dimension_input': was expecting 'JSON_NUMBER'", root_it->name());
            dimension_input_ = static_cast<unsigned int>(root_it->as_int());
        }
        
        // Allocate memory And Read data
        if (bimodal_) {
            data[0] = reallocate<float>(data[0],
                                        max_length_ * dimension_input_,
                                        length_ * dimension_input_);
            data[1] = reallocate<float>(data[1],
                                        max_length_ * (dimension_ - dimension_input_),
                                        length_ * (dimension_ - dimension_input_));
            root_it = root.find("data_input");
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->type() != JSON_ARRAY)
                throw JSONException("Wrong type for node 'data_input': was expecting 'JSON_ARRAY'", root_it->name());
            json2array(*root_it, data[0], length_ * dimension_input_);
            
            root_it = root.find("data_output");
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->type() != JSON_ARRAY)
                throw JSONException("Wrong type for node 'data_output': was expecting 'JSON_ARRAY'", root_it->name());
            json2array(*root_it, data[1], length_ * (dimension_ - dimension_input_));
        } else {
            data[0] = reallocate<float>(data[0],
                                        max_length_ * dimension_,
                                        length_ * dimension_);
            root_it = root.find("data");
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->type() != JSON_ARRAY)
                throw JSONException("Wrong type for node 'data': was expecting 'JSON_ARRAY'", root_it->name());
            json2array(*root_it, data[0], length_ * dimension_);
        }
        
        max_length_ = length_;
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (std::exception &e) {
        throw JSONException(e, root.name());
    }
}

#pragma mark Moments
std::vector<float> xmm::Phrase::mean() const
{
    std::vector<float> mean(dimension_);
    for (unsigned int d=0; d<dimension_; d++) {
        mean[d] = 0.;
        for (unsigned int t=0; t<length_; t++) {
            mean[d] += operator()(t, d);
        }
        mean[d] /= float(length_);
    }
    return mean;
}

std::vector<float> xmm::Phrase::variance() const
{
    std::vector<float> variance(dimension_);
    std::vector<float> _mean = mean();
    for (unsigned int d=0; d<dimension_; d++) {
        variance[d] = 0.;
        for (unsigned int t=0; t<length_; t++) {
            variance[d] += (operator()(t, d) - _mean[d]) * (operator()(t, d) - _mean[d]);
        }
        variance[d] /= float(length_);
    }
    return variance;
}

