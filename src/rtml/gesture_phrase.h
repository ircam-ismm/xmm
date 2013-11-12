//
// gesture_phrase.h
//
// Template class for Gesture data phrase
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#ifndef mhmm_gesture_phrase_h
#define mhmm_gesture_phrase_h

#include "phrase.h"

using namespace std;

namespace momos {
    const int GPHRASE_DEFAULT_DIMENSION_GESTURE = 1;
    
    /*!
     @class GesturePhrase
     @brief Unimodal Gesture phrase
     @tparam ownData Defines if the data is stored in the Phrase or shared with another container
     */
    template <bool ownData>
    class GesturePhrase : public Phrase<ownData, 1> {
    public:
#pragma mark -
#pragma mark Constructors
        /*! @name Constructors */
        /*!
         Class Constructor
         @param _dimension_gesture dimension of the gesture stream
         */
        GesturePhrase(int _dimension_gesture=GPHRASE_DEFAULT_DIMENSION_GESTURE)
        : Phrase<ownData, 1>(NULL)
        {
            this->set_dimension(_dimension_gesture);
        }
        
#pragma mark -
#pragma mark Connect (shared data)
        /*! @name Connect (shared data) */
        /*!
         @brief Connect the phrase to a shared container
         
         This method is only usable in Shared Memory (ownData=false)
         @param _data_gesture pointer to the gesture data array
         @param _length length of the data array
         @throw runtime_error if phrase has own Data
         */
        
        void connect(float *_data_gesture,
                     int _length)
        {
            float* _data[1] = {_data_gesture};
            Phrase<ownData, 1>::connect(_data, _length);
        }
        
    };
}

#endif
