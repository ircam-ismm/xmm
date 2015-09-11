/*
 * phrase.h
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

#ifndef xmm_lib_phrase_h_
#define xmm_lib_phrase_h_

#include <cmath>
#include "json_utilities.h"
#include "xmm_common.h"
#include "label.h"

namespace xmm
{
    /**
     * @defgroup TrainingSet Training Datasets
     */
    
    /**
     * @ingroup TrainingSet
     * @class Phrase
     * @brief Data phrase
     * @details The Phrase class can be used to store unimodal and Bimodal data phrases.
     * It can have an autonomous memory, or this memory can be shared with another data
     * container. This can be specified by using the 'SHARED_MEMORY' flag in the constructor.
     * The phrase can be either unimodal - i.e. it contains a single 2D array to store the data, -
     * or bimodal - i.e. it contains 2 arrays to store the input and output modalities. The latter case
     * can be specified by using the 'BIMODAL' flag in the constructor.
     */
    class Phrase : public Writable
    {
    public:
        ///@cond DEVDOC
        
        static const unsigned int DEFAULT_DIMENSION = 1;
        static const unsigned int ALLOC_BLOCKSIZE = 256;
        
        ///@endcond
        
#pragma mark -
#pragma mark === Public Interface ===
#pragma mark > Constructors
        /** @name Constructors */
        ///@{
        
        /**
         * @brief Phrase Constructor
         *
         * @param dimension total dimension of the data phrase
         * @param flags construction flags: SHARED_MEMORY defines a phrase that cannot edit its memory. data
         * is therefore passed by pointer and stored in another container. BIMODAL defines if the phrase has 2 modalities
         * @param dimension_input dimension of input modality in the case of a bimodal phrase
         */
        Phrase(xmm_flags flags=NONE,
               unsigned int dimension=DEFAULT_DIMENSION,
               unsigned int dimension_input = 0);
        
        /**
         * @brief Copy Constructor
         * @param src source Phrase
         */
        Phrase(Phrase const& src);
        
        /**
         * @brief Assignment
         * @param src source Phrase
         */
        Phrase& operator=(Phrase const& src);
        
        /**
         * @brief Destructor.
         * @details Data is only deleted if the memory is not shared (no SHARED_MEMORY flag)
         */
        virtual ~Phrase();
        
        ///@}
        
#pragma mark > Tests
        /** @name Tests */
        ///@{
        
        /**
         * @brief Checks if the phrase is empty (length 0)
         */
        bool is_empty() const;
        
        /**
         * @brief Check equality
         * @warning 2 phrases are considered equal iff their data have the same address + same length
         * @param src source Phrase
         */
        bool operator==(Phrase const& src);
        
        /**
         * @brief Check inequality
         * @see operator==
         * @param src source Phrase
         */
        bool operator!=(Phrase const& src);
        
        ///@}
        
#pragma mark > Accessors
        /** @name Accessors */
        ///@{
        
        /**
         * @return length of the phrase
         */
        unsigned int length() const;
        
        /**
         * @brief trim phrase to specific length phrase length
         * @warning only works if trimed to shorter phrase
         */
        void trim(unsigned int phraseLength);
        
        /**
         * @brief trim phrase to minimal length of modalities
         */
        void trim();
        
        /**
         * @return total dimension of the data
         */
        unsigned int dimension() const;
        
        /**
         * @return dimension of the input modality
         */
        unsigned int dimension_input() const;
        
        /**
         * @return dimension of the output modality
         */
        unsigned int dimension_output() const;
        
        /**
         * Set total dimension
         * @param dimension target dimension
         * @throws domain_error if dimension < 1
         */
        void set_dimension(unsigned int dimension);
        
        /**
         * Set dimension of the input modality
         * @param dimension_input target dimension
         * @throws runtime_error if the phrase is not bimodal
         * @throws invalid_argument if The dimension of the input modality exceeds the total dimension
         */
        void set_dimension_input(unsigned int dimension_input);
        
        ///@}
        
#pragma mark > Connect (shared data)
        /** @name Connect (shared data) */
        ///@{
        
        /**
         * @brief Connect a unimodal phrase to a shared container
         * @warning This method is only usable in Shared Memory (construction with SHARED_MEMORY flag)
         * @param pointer_to_data pointer to the data array
         * @param length length of the data array
         * @throws runtime_error if phrase has own Data
         */
        void connect(float *pointer_to_data, unsigned int length);
        
        /**
         * @brief Connect a Bimodal phrase to a shared container
         * @warning This method is only usable in Shared Memory (construction with SHARED_MEMORY flag)
         * @param pointer_to_data_input pointer to the data array of the input modality
         * @param pointer_to_data_output pointer to the data array of the output modality
         * @param length length of the data array
         * @throws runtime_error if phrase has own Data
         */
        void connect(float *pointer_to_data_input, float *pointer_to_data_output, unsigned int length);
        
        /**
         * @brief Connect a Bimodal phrase to a shared container for the input modality
         * @warning This method is only usable in Shared Memory (construction with SHARED_MEMORY flag)
         * @param pointer_to_data pointer to the data array of the input modality
         * @param length length of the data array
         * @throws runtime_error if phrase has own Data
         */
        void connect_input(float *pointer_to_data, unsigned int length);
        
        /**
         * @brief Connect a Bimodal phrase to a shared container for the output modality
         * @warning This method is only usable in Shared Memory (construction with SHARED_MEMORY flag)
         * @param pointer_to_data pointer to the data array of the output modality
         * @param length length of the data array
         * @throws runtime_error if phrase has own Data
         */
        void connect_output(float *pointer_to_data, unsigned int length);
        
        /**
         * @brief Disconnect a phrase from a shared container
         * @warning This method is only usable in Shared Memory (construction with SHARED_MEMORY flag)
         * @throws runtime_error if phrase has own Data
         */
        void disconnect();
        
        ///@}
        
#pragma mark > Record (own Data)
        /** @name Record (own Data) */
        ///@{
        
        /**
         * @brief Record observation
         * @details Appends the observation vector observation to the data array.\n
         * This method is only usable in Own Memory (no SHARED_MEMORY flag)
         * @param observation observation vector (C-like array which must have the size of the total
         * dimension of the data across all modalities)
         * @throws runtime_error if data is shared (construction with SHARED_MEMORY flag)
         */
        void record(std::vector<float> const& observation);
        
        /**
         * @brief Record observation on input modality
         * Appends the observation vector observation to the data array\n
         * This method is only usable in Own Memory (no SHARED_MEMORY flag)
         * @param observation observation vector (C-like array which must have the size of the total
         * dimension of the data across all modalities)
         * @throws runtime_error if data is shared (ownData == false)
         */
        void record_input(std::vector<float> const& observation);
        
        /**
         * @brief Record observation on output modality
         * Appends the observation vector observation to the data array\n
         * This method is only usable in Own Memory (no SHARED_MEMORY flag)
         * @param observation observation vector (C-like array which must have the size of the total
         * dimension of the data across all modalities)
         * @throws runtime_error if data is shared (construction with SHARED_MEMORY flag)
         */
        void record_output(std::vector<float> const& observation);
        
        /**
         * @brief Reset length of the phrase to 0 ==> empty phrase\n
         * @warning the memory is not released (only done in destructor).
         * @throws runtime_error if data is shared (construction with SHARED_MEMORY flag)
         */
        void clear();
        
        ///@}
        
#pragma mark > Access Data
        /** @name Access Data */
        ///@{
        
        /**
         * @brief Access data at a given time index and dimension.
         * @param index time index
         * @param dim dimension considered, indexed from 0 to the total dimension of the data across modalities
         * @throws out_of_range if time index or dimension are out of bounds
         */
        float at(unsigned int index, unsigned int dim) const;
        
        /**
         * @brief Access data at a given time index and dimension.
         * @param index time index
         * @param dim dimension considered, indexed from 0 to the total dimension of the data across modalities
         * @throws out_of_range if time index or dimension are out of bounds
         */
        float operator()(unsigned int index, unsigned int dim) const;
        
        /**
         * @brief Get pointer to the data at a given time index
         * @param index time index
         * @warning  this method can be used only for unimodal phrases (no BIMODAL flag)
         * @throws out_of_range if time index is out of bounds
         * @throws runtime_error if the phrase is bimodal
         * @return pointer to the data array of the modality, for the given time index
         */
        float* get_dataPointer(unsigned int index) const;
        
        /**
         * @brief Get pointer to the data at a given time index for the input modality
         * @warning  this method can be used only for bimodal phrases (construction with BIMODAL flag)
         * @param index time index
         * @throws out_of_range if time index is out of bounds
         * @throws runtime_error if the phrase is unimodal
         * @return pointer to the data array of the modality, for the given time index
         */
        float* get_dataPointer_input(unsigned int index) const;
        
        /**
         * @brief Get pointer to the data at a given time index for the output modality
         * @warning  this method can be used only for bimodal phrases (construction with BIMODAL flag)
         * @param index time index
         * @throws out_of_range if time index is out of bounds
         * @throws runtime_error if the phrase is unimodal
         * @return pointer to the data array of the modality, for the given time index
         */
        float* get_dataPointer_output(unsigned int index) const;
        
        ///@}
        
#pragma mark > JSON I/O
        /** @name JSON I/O  */
        ///@{
        
        /**
         * @brief Write to JSON Node
         * @return JSON Node containing phrase information
         */
        JSONNode to_json() const;
        
        /**
         * @brief Read from JSON Node
         * @param root JSON Node containing phrase information
         * @throws JSONException if the JSON Node has a wrong format
         */
        void from_json(JSONNode root);
        
        ///@}
        
#pragma mark > Moments
        /** @name Moments */
        ///@{
        
        /**
         * @brief Compute the mean of the data phrase along the time axis
         * @return mean of the phrase (along time axis, full-size)
         */
        std::vector<float> mean() const;
        
        /**
         * @brief Compute the variance of the data phrase along the time axis
         * @return variance of the phrase (along time axis, full-size)
         */
        std::vector<float> variance() const;
        
        ///@}
        
#pragma mark -
#pragma mark === Public Attributes ===
        /** @name Public Attributes */
        ///@{
        
        /**
         * @brief labels of the columns of the phrase (e.g. descriptor names)
         */
        std::vector<std::string> column_names_;
        
        ///@}
        
    private:
        ///@cond DEVDOC
        
#pragma mark -
#pragma mark === Private Methods ===
        /** @name utilities (protected) */
        ///@{
        
        /**
         Copy from a Phrase (called by copy constructor and assignment)
         */
        void _copy(Phrase *dst, Phrase const& src);
        
        /**
         * @brief Memory Allocation
         * @details used record mode (no SHARED_MEMORY flag), the data vector is reallocated
         * with a block size ALLOC_BLOCKSIZE
         */
        void reallocate_length();
        
        ///@}
        
#pragma mark -
#pragma mark === Private Attributes ===
        /**
         * @brief Defines if the phrase stores the data itself.
         */
        bool owns_data_;
        
        /**
         * @brief Defines if the phrase is bimodal (true) or unimodal (false)
         */
        bool bimodal_;
        
        /**
         * @brief true if the phrase does not contain any data
         */
        bool is_empty_;
        
        /**
         * @brief Length of the phrase. If bimodal, it is the minimal length between modalities
         */
        unsigned int length_;
        
        /**
         * @brief Length of the array of the input modality
         */
        unsigned int length_input_;
        
        /**
         * @brief Length of the array of the output modality
         */
        unsigned int length_output_;
        
        /**
         * @brief Allocated length (only used in own memory mode)
         */
        unsigned int max_length_;
        
        /**
         * @brief Total dimension of the phrase
         */
        unsigned int dimension_;
        
        /**
         * @brief Used in bimodal mode: dimension of the input modality.
         */
        unsigned int dimension_input_;
        
        /**
         * @brief Pointer to the Data arrays
         * @details data has a size 1 in unimodal mode, 2 in bimodal mode.
         */
        float **data;
        
        ///@endcond
    };
    
#pragma mark -
#pragma mark Utility: Memory Allocation
    /**
     * @brief Reallocate a C-like array (using c++ std::copy)
     * @param src source array
     * @param dim_src initial dimension
     * @param dim_dst target dimension
     * @return resized array (content is conserved)
     */
    template <typename T>
    T* reallocate(T *src, unsigned int dim_src, unsigned int dim_dst) {
        T *dst = new T[dim_dst];
        
        if (!src) return dst;
        
        if (dim_dst > dim_src) {
            std::copy(src, src+dim_src, dst);
        } else {
            std::copy(src, src+dim_dst, dst);
        }
        delete[] src;
        return dst;
    }
}

#endif

