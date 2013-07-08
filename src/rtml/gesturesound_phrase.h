//
//  gesturesound_phrase.h
//  rtml
//
//  Created by Jules Francoise on 20/01/13.
//
//

#ifndef rtml_gesturesound_phrase_h
#define rtml_gesturesound_phrase_h

#include "phrase.h"

#define MPHRASE_DEFAULT_DIMENSION_GESTURE 1
#define MPHRASE_DEFAULT_DIMENSION_SOUND 1

using namespace std;

/*!
 @class GestureSoundPhrase
 @brief Multimodal Gesture-Sound phrase
 @tparam ownData Defines if the data is stored in the Phrase or shared with another container
 */
template <bool ownData>
class GestureSoundPhrase : public Phrase<ownData, 2> {
public:
#pragma mark -
#pragma mark Constructors
    /*! @name Constructors */
    /*!
     Class Constructor
     @param _dimension_gesture dimension of the gesture stream
     @param _dimension_sound dimension of the sound parameter stream
     */
    GestureSoundPhrase(int _dimension_gesture=MPHRASE_DEFAULT_DIMENSION_GESTURE,
                       int _dimension_sound=MPHRASE_DEFAULT_DIMENSION_SOUND)
    : Phrase<ownData, 2>(NULL)
    {
        this->set_dimension_gesture(_dimension_gesture);
        this->set_dimension_sound(_dimension_sound);
    }
    
#pragma mark -
#pragma mark Connect (shared data)
    /*! @name Connect (shared data) */
    /*!
     @brief Connect the phrase to a shared container
     
     This method is only usable in Shared Memory (ownData=false)
     @param _data_gesture pointer to the gesture data array
     @param _data_sound pointer to the sound parameters data array
     @param _length length of the data array
     @throw runtime_error if phrase has own Data
     */
    
    void connect(float *_data_gesture,
                 float *_data_sound,
                 int _length)
    {
        float* _data[2] = {_data_gesture, _data_sound};
        Phrase<ownData, 2>::connect(_data, _length);
    }
    
#pragma mark -
#pragma mark Access Data
    /*! @name Access Data */
    /*!
     Get pointer to the gesture array for a given time index
     @param timeIndex time index
     */
    float *get_dataPointer_gesture(int timeIndex) const
    {
        return this->get_dataPointer(timeIndex, 0);
    }
    
    /*!
     Get pointer to the sound parameters array for a given time index
     @param timeIndex time index
     */
    float *get_dataPointer_sound(int timeIndex) const
    {
        return this->get_dataPointer(timeIndex, 1);
    }
    
#pragma mark -
#pragma mark Accessors
    /*! @name Accessors */
    int get_dimension_gesture() const
    {
        return this->get_dimension(0);
    }
    
    int get_dimension_sound() const
    {
        return this->get_dimension(1);
    }
    
    void set_dimension_gesture(int _dimension_gesture)
    {
        this->set_dimension(_dimension_gesture, 0);
    }
    
    void set_dimension_sound(int _dimension_sound)
    {
        this->set_dimension(_dimension_sound, 1);
    }
};

#endif
