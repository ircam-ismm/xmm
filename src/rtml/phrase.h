//
//  phrase.h
//  rtml
//
//  Created by Jules Francoise on 20/01/13.
//
//

#ifndef __rtml__phrase__
#define __rtml__phrase__

#include <iostream>
#include <ostream>
#include <istream>
#include <sstream>
#include <rtmlexception.h>
#include "utility.h"

using namespace std;

const int PHRASE_DEFAULT_DIMENSION = 1;
const int PHRASE_ALLOC_BLOCKSIZE = 256;

/*!
 @class Phrase
 @brief Multimodal data phrase
 @todo class description
 @todo re-introduce datatype as a template parameter => handle discrete data types
 @tparam ownData Defines if the data is stored in the Phrase or shared with another container
 @tparam nbModalities number of modalities
 */
template <bool ownData=true, unsigned int nbModalities=1>
class Phrase {
public:
    float *data[nbModalities];
    
#pragma mark -
#pragma mark Constructors
    /*! @name Constructors */
    /*!
     Class Constructor
     @param _dimension dimensions of the data (Default=PHRASE_DEFAULT_DIMENSION)
     */
    Phrase(int _dimension[nbModalities]=NULL)
    {
        for (unsigned int modality=0; modality<nbModalities; modality++) {
            dimension[modality] = _dimension ? _dimension[modality] : PHRASE_DEFAULT_DIMENSION;
            data[modality] = NULL;
        }
        length = 0;
        max_length = 0;
        empty = true;
    }
    
    /*!
     Copy Constructor
     */
    Phrase(Phrase<ownData, nbModalities> const& src)
    {
        for (unsigned int modality=0; modality<nbModalities; modality++) {
            dimension[modality] = PHRASE_DEFAULT_DIMENSION;
            data[modality] = NULL;
        }
        _copy(this, src);
    }
    
    /*!
     Assignment
     */
    virtual Phrase& operator=(Phrase<ownData, nbModalities> const& src)
    {
        if(this != &src)
        {
            _copy(this, src);
            
        }
        return *this;
    }
    
    /*!
     Copy from a Phrase (called by copy constructor and assignment)
     */
    virtual void _copy(Phrase<ownData, nbModalities> *dst, Phrase<ownData, nbModalities> const& src)
    {
        dst->max_length = src.max_length;
        dst->length = src.length;
        dst->empty = src.empty;
        
        for (unsigned int modality=0; modality<nbModalities; modality++)
        {
            dst->dimension[modality] = src.dimension[modality];
            if (ownData)
            {
                if (dst->data[modality]) {
                    delete[] dst->data[modality];
                    dst->data[modality] = NULL;
                }
                if (dst->max_length > 0) {
                    dst->data[modality] = new float[dst->max_length*dst->dimension[modality]];
                    copy(src.data[modality], src.data[modality]+dst->length*dst->dimension[modality], dst->data[modality]);
                }
            }
            else
            {
                dst->data[modality] = src.data[modality];
            }
        }
    }
    
    /*!
     Destructor.\n
     Data is only deleted if the memory is not shared (ownData=true)
     */
    virtual ~Phrase()
    {
        if (ownData) {
            for (unsigned int modality=0; modality<nbModalities; modality++) {
                delete[] data[modality];
                data[modality] = NULL;
            }
        }
    }
    
#pragma mark -
#pragma mark Tests
    /*! @name Tests */
    /*!
     Checks if the phrase is empty (length 0)
     */
    bool is_empty() const
    {
        return empty;
    }
    
    /*!
     Check equality
     @warning 2 phrases are considered equal iff their data have the same address + same length
     */
    bool operator==(Phrase<ownData, nbModalities> const& src)
    {
        if (length != src.length) return false;
        for (unsigned int modality=0; modality<nbModalities; modality++) {
            if (data[modality] != src.data[modality] ||
                dimension[modality] != src.dimension[modality])
                return false;
        }
        return true;
    }
    
    bool operator!=(Phrase<ownData, nbModalities> const& src)
    {
        return !(operator==(src));
    }
    
#pragma mark -
#pragma mark Accessors
    /*! @name Accessors */
    /*!
     @return length of the phrase
     */
    int getlength() const
    {
        return length;
    }
    
    /*!
     @return dimension of a given modality
     @todo unsigned would be better
     @param modality index of the modality to consider
     */
    int get_dimension(int modality=0) const
    {
        return dimension[modality];
    }
    
    /*!
     Set dimension of a given modality.
     @param _dimension target dimension
     @param modality index of the modality to consider
     @throw RTMLException if _dimension < 1
     */
    void set_dimension(int _dimension, int modality=0)
    {
        if (_dimension < 1) throw RTMLException("Dimension must be striclty positive", __FILE__, __FUNCTION__, __LINE__);
        if (_dimension == dimension[modality]) return;
        
        if (ownData) {
            data[modality] = reallocate(data[modality],
                                        max_length*dimension[modality],
                                        max_length*_dimension);
        }
        dimension[modality] = _dimension;
    }
    
#pragma mark -
#pragma mark Connect (shared data)
    /*! @name Connect (shared data) */
    /*!
     @brief Connect a phrase to a shared container
     
     This method is only usable in Shared Memory (ownData=false)
     @param _data array of pointers to the data (nbModalities C-like Array)
     @param _length length of the data array (dimension can only be set via the accessor)
     @throw RTMLException if phrase has own Data
     */
    void connect(float *_data[nbModalities],
                 int _length)
    {
        if (ownData) throw RTMLException("Cannot connect a phrase with own data", __FILE__, __FUNCTION__, __LINE__);
        
        for (unsigned int modality=0; modality<nbModalities; modality++) {
            data[modality] = _data[modality];
        }
        length = _length;
        empty = false;
    }
    
    /*!
     @brief Disconnect a phrase from a shared container
     @throw RTMLException if phrase has own Data
     */
    void disconnect()
    {
        if (ownData) throw RTMLException("Cannot disconnect a phrase with own data", __FILE__, __FUNCTION__, __LINE__);
        
        for (unsigned int modality=0; modality<nbModalities; modality++) {
            data[modality] = NULL;
        }
        length = 0;
        empty = true;
    }
    
#pragma mark -
#pragma mark Record (Own Data)
    /*! @name Record (Own Data) */
    /*!
     @brief Record observation
     
     Appends the observation vector observation to the data array\n
     This method is only usable in Own Memory (ownData=true)
     @param observation observation vector (C-like array which must have the size of the total
     dimension of the data across all modalities)
     @throw RTMLException if data is shared (ownData == false)
     */
    void record(float *observation)
    {
        if (!ownData) throw RTMLException("Cannot record in shared data phrase", __FILE__, __FUNCTION__, __LINE__);
        
        if (length >= max_length || max_length == 0) {
            reallocate_length();
        }
        
        int dim_offset(0);
        for (unsigned int modality=0; modality<nbModalities; modality++) {
            for (int d=0 ; d<dimension[modality] ; d++)
                data[modality][length * dimension[modality] + d] = observation[dim_offset+d];
            dim_offset += dimension[modality];
        }
        length++;
        empty = false;
    }
    
    /*!
     Memory Allocation: in record mode (OwnData=true), the data vector is reallocated with a block size PHRASE_ALLOC_BLOCKSIZE
     */
    void reallocate_length()
    {
        for (unsigned int modality=0; modality<nbModalities; modality++) {
            data[modality] = reallocate<float>(data[modality],
                                               max_length*dimension[modality],
                                               (max_length+PHRASE_ALLOC_BLOCKSIZE)*dimension[modality]);
        }
        max_length += PHRASE_ALLOC_BLOCKSIZE;
    }
    
    /*!
     Reset length of the phrase to 0 ==> empty phrase\n
     @warning the memory is not released (only done in destructor).
     @throw RTMLException if data is shared (ownData == false)
     */
    void clear()
    {
        if (!ownData) throw RTMLException("Cannot clear a shared data phrase", __FILE__, __FUNCTION__, __LINE__);
        
        length = 0;
        empty = true;
    }
    
#pragma mark -
#pragma mark Access Data
    /*! @name Access Data */
    /*!
     Access data at a given time index and dimension.
     @param index time index
     @param dim dimension considered, indexed from 0 to the total dimension of the data across modalities
     @throw RTMLException if time index or dimension are out of bounds
     */
    float operator()(int index, int dim) const
    {
        if (index >= length) throw RTMLException("Phrase: index out of bounds", __FILE__, __FUNCTION__, __LINE__);
        
        for (unsigned int modality=0; modality<nbModalities; modality++) {
            if (dim < dimension[modality]) {
                return data[modality][index * dimension[modality] + dim];
            }
            dim -= dimension[modality];
        }
        
        throw RTMLException("Phrase: dimension out of bounds", __FILE__, __FUNCTION__, __LINE__);
    }
    
    /*!
     Get pointer to the data a modality at a given time index
     @param index time index
     @param modality index of the modality
     @throw RTMLException if time index is out of bounds
     @return pointer to the data array of the modality, for the given time index
     */
    float* get_dataPointer(int index, int modality=0) const
    {
        if (index >= length) throw RTMLException("Phrase: index out of bounds", __FILE__, __FUNCTION__, __LINE__);
        
        return data[modality] + index * dimension[modality];
    }
    
#pragma mark -
#pragma mark File IO / Stream IO
    /*! @name File IO  */
    /*!
     Write Data to Stream
     @param outStream output stream
     */
    void write(ostream& outStream) const
    {
        outStream << "# Phrase\n";
        outStream << "# Number of modalities\n";
        outStream << nbModalities << endl;
        outStream << "# Length\n";
        outStream << length << endl;
        outStream << "# Dimensions\n";
        for (unsigned int modality=0; modality<nbModalities; modality++) {
            outStream << dimension[modality] << " ";
        }
        outStream << endl;
        
        outStream << "# Data\n";
        for (int t=0 ; t<length ; t++) {
            for (unsigned int modality=0; modality<nbModalities; modality++)
                for (int d=0; d<dimension[modality]; d++)
                    outStream << data[modality][t*dimension[modality]+d] << " ";
            outStream << endl;
        }
    }
    
    /*!
     Read Data from Stream
     @param inStream input stream
     @throw RTMLException if the format doesn't match, or phrase has shared data
     */
    void read(istream& inStream)
    {
        if (!ownData) {
            throw RTMLException("Phrase: cannot read phrase with shared data", __FILE__, __FUNCTION__, __LINE__);
        }
        
        // Read Number of modalities
        skipComments(&inStream);
        int nbModalities_;
        inStream >> nbModalities_;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        if (nbModalities_ != nbModalities)
            throw RTMLException("Number of modalities mismatch: Phrase is not of the same type", __FILE__, __FUNCTION__, __LINE__);
        
        // Read Length of training phrase
        skipComments(&inStream);
        int length_;
        inStream >> length_;
        
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        // Read Dimensions
        skipComments(&inStream);
        for (unsigned int modality=0; modality < nbModalities; modality++) {
            int dim;
            inStream >> dim;
            if (!inStream.good())
                throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
            
            set_dimension(dim, modality);
        }
        
        clear();
        
        // Read Data
        skipComments(&inStream);
        
        int dimension_total(0);
        for (unsigned int modality=0; modality<nbModalities; modality++)
            dimension_total += dimension[modality];
        
        float *obs = new float[dimension_total];
        for (int t=0; t<length_; t++) {
            for (int d=0; d<dimension_total; d++) {
                inStream >> obs[d];
                if (!inStream.good())
                    throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
            }
            record(obs);
        }
        
        if (length_ == 0) {
            for (unsigned int m=0; m<nbModalities; m++) {
                data[m] = NULL;
            }
        }
    }
    
#pragma mark -
#pragma mark Moments
    /*! @name Moments */
    /*!
     @return mean of the phrase for each modality
     */
    vector<float> mean() const
    {
        int dimension_total(0);
        for (unsigned int modality=0; modality<nbModalities; modality++)
            dimension_total += dimension[modality];
        
        vector<float> mean(dimension_total);
        for (int d=0; d<dimension_total; d++) {
            mean[d] = 0.;
            for (int t=0; t<length; t++) {
                mean[d] += operator()(t, d);
            }
            mean[d] /= float(length);
        }
        return mean;
    }
    
    /*!
     @return variance of the phrase for each modality
     */
    vector<float> variance() const
    {
        int dimension_total(0);
        for (unsigned int modality=0; modality<nbModalities; modality++)
            dimension_total += dimension[modality];
        
        vector<float> variance(dimension_total);
        vector<float> _mean = mean();
        for (int d=0; d<dimension_total; d++) {
            variance[d] = 0.;
            for (int t=0; t<length; t++) {
                variance[d] += pow(operator()(t, d) - _mean[d], 2);
            }
            variance[d] /= float(length);
        }
        return variance;
    }
    
#pragma mark -
#pragma mark Python
#ifdef SWIGPYTHON
    /*!@name Python utility */
    /*!
     @brief Record observation
     
     Appends the observation vector observation to the data array\n
     This method is only usable in Own Memory (ownData=true)
     @param observation observation vector (C-like array which must have the size of the total
     dimension of the data across all modalities)
     @throw RTMLException if data is shared (ownData == false)
     @todo check if can use float* in swig
     @todo maybe can be integrated in swig interface file
     */
    void record(int dimension_total, double *observation)
    {
        float *observation_float = new float[dimension_total];
        for (int d=0; d<dimension_total; d++) {
            observation_float[d] = float(observation[d]);
        }
        
        record(observation_float);
        
        delete[] observation_float;
    }
#endif
    /*!@name*/
protected:
    bool empty;
    int length;
    int max_length;
    int dimension[nbModalities];
};

#endif

