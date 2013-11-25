//
//  markov_phrase.h
//  mhmm
//
//  Created by Jules Francoise on 25/04/13.
//
//

#ifndef mhmm_markov_phrase_h
#define mhmm_markov_phrase_h

#define HPHRASE_DEFAULT_DIMENSION_GESTURE 1

#include "phrase.h"

using namespace std;
using namespace rtml;

template <bool ownData>
class HierarchicalGesturePhrase : public Phrase<ownData, 1> {
public:
#pragma mark -
#pragma mark Constructors
    HierarchicalGesturePhrase(int _dimension_gesture=HPHRASE_DEFAULT_DIMENSION_GESTURE)
    : Phrase<ownData, 1>(NULL)
    {
        this->set_dimension_gesture(_dimension_gesture);
    }
    
    void reallocate_length()
    {
        Phrase<ownData, 1>::reallocate_length();
        exitprob.resize(this->max_length+PHRASE_ALLOC_BLOCKSIZE);
    }
    
#pragma mark -
#pragma mark Connect (shared data)
    void connect(float *_data_gesture,
                 int _length)
    {
        float* _data[1] = {_data_gesture};
        Phrase<ownData, 1>::connect(_data, _length);
        exitprob.resize(_length);
    }
    
#pragma mark -
#pragma mark Access Data
    float *get_dataPointer_gesture(int timeIndex) const
    {
        return this->get_dataPointer(timeIndex, 0);
    }
    
#pragma mark -
#pragma mark Accessors
    int get_dimension_gesture() const
    {
        return this->get_dimension(0);
    }
    
    void set_dimension_gesture(int _dimension_gesture)
    {
        this->set_dimension(_dimension_gesture, 0);
    }
    
#pragma mark -
#pragma mark Exit Probabilities
    void setExitEnd(int exitlength, double *exitvec)
    {
        if (this->length == 0)
            throw runtime_error("Phrase is empty");
        
        for (int t=this->length-exitlength, i=0; t<this->length; t++, i++) {
            exitprob[t] = exitvec[i];
        }
    }
    
	void setExitUniform(double p)
    {
        if (this->length == 0)
            throw runtime_error("Phrase is empty");
        
        for (int t=0, i=0; t<this->length; t++, i++) {
            exitprob[t] = p;
        }
    }
    
	void addExitPoint(int time, double p)
    {
        if (this->length == 0)
            throw runtime_error("Phrase is empty");
        if (time >= this->length)
            throw runtime_error("Time index out of bounds");;
        
        exitprob[time] = p;
    }
    
	void addExitSegment(int t1, int t2, double p)
    {
        if (this->length == 0)
            throw runtime_error("Phrase is empty");
        if (t2 >= this->length)
            throw runtime_error("Time index out of bounds");;
        
        for (int t=t1; t<=t2; t++) {
            exitprob[t] = p;
        }
    }
    
protected:
	vector<float>  exitprob;
};

#endif
