//
//  mubu_training_set.h
//  rtml
//
//  Created by Jules Francoise on 21/01/13.
//
//

#ifndef rtml_mubu_training_set_h
#define rtml_mubu_training_set_h

#include "training_set.h"
extern "C" {
#include "mubuext.h"
}

using namespace std;

#pragma mark -
#pragma mark Prototypes
static t_max_err notifyMethod(MubuTrainingSet *self, t_symbol *s, t_symbol *msg, void *sender, void *data);
static void referMethod(MubuTrainingSet *self, t_symbol *s, short ac, t_atom *at);
void updateTrainingSet(MubuTrainingSet *self);

#pragma mark -
#pragma mark Class Definition
template <typename phraseType, typename labelType=int>
struct MubuTrainingSet : public TrainingSet<phraseType, labelType> {
public:
    MubuTrainingSet(t_object *_maxObj, LearningModel<phraseType, labelType>* _parent=NULL);
    ~MubuTrainingSet();
    
    t_max_err notify(t_symbol *msg, void *sender, void *data);
    void refer();
    void setContainerName(t_symbol *name);
    
    void updateTrainingSet();
    
protected:
    t_symbol *containerName;
    MuBuT *container;
    
    int trackIndex_gesture;
    int trackIndex_sound;
    
    MuBuTrkT **TrainingTracks_gesture;
    MuBuTrkT **TrainingTracks_sound;
    float **TrainingData_gesture;
    float **TrainingData_sound;
    
    t_systhread_mutex  mutex;
    void *updateQelem;
    
    int autoUpdate;
    t_object *maxObj;
    t_atom *attr_out;
};

template <typename phraseType, typename labelType>
MubuTrainingSet<phraseType, labelType>::MubuTrainingSet(t_object *_maxObj, LearningModel<phraseType, labelType>* _parent)
: TrainingSet<phraseType, labelType>(_parent)
{
    maxObj = _maxObj;
    
    containerName = NULL;
    container = NULL;
    
    trackIndex_gesture = -1;
    trackIndex_sound = -1;
    
    TrainingTracks_gesture = NULL;
    TrainingTracks_sound = NULL;
    TrainingData_gesture = NULL;
    TrainingData_sound = NULL;
    
    systhread_mutex_new(&mutex, 0);
    autoUpdate = 0;
    
    updateQelem = qelem_new(maxObj, (method)updateTrainingSet);
}

template <typename phraseType, typename labelType>
MubuTrainingSet<phraseType, labelType>::~MubuTrainingSet()
{
    mubuRef_set(&container, (t_object *)maxObj, NULL);
    systhread_mutex_free(&mutex);
    
    if(TrainingTracks_gesture != NULL)
        sysmem_freeptr(TrainingTracks_gesture);
    if(TrainingTracks_sound != NULL)
        sysmem_freeptr(TrainingTracks_sound);
    if(TrainingData_gesture != NULL)
        sysmem_freeptr(TrainingData_gesture);
    if(TrainingData_sound != NULL)
        sysmem_freeptr(TrainingData_sound);
}

template <typename phraseType, typename labelType>
t_max_err MubuTrainingSet<phraseType, labelType>::notify(t_symbol *msg, void *sender, void *data)
{
    systhread_mutex_lock(mutex);
    
    cout << "notify: msg = " << msg->s_name << endl;
    
    /* handle change of mubu/imubu bound to name */
    if(msg == gensym("globalsymbol_binding"))
        mubuRef_connect(&container, (t_object *)maxObj, containerName);
    else if(msg == gensym("globalsymbol_unbinding"))
        mubuRef_connect(&container, (t_object *)maxObj, NULL);
    
    if(autoUpdate &&sender == container)
    {
        qelem_front(updateQelem);
    }
    
    systhread_mutex_unlock(mutex);
    
    return MAX_ERR_NONE;
}

template <typename phraseType, typename labelType>
void MubuTrainingSet<phraseType, labelType>::setContainerName(t_symbol *name)
{
    systhread_mutex_lock(mutex);
    
    if(name != containerName)
    {
        containerName = name;
        mubuRef_set(&container, (t_object *)maxObj, name);
        
        if (autoUpdate)
            qelem_front(updateQelem);
    }
    
    systhread_mutex_unlock(mutex);
}

template <typename phraseType, typename labelType>
int MubuTrainingSet_init(t_class *c, MubuTrainingSet<phraseType, labelType> *mts)
{
    MuBuErrorT err = mubuExt_init();
    static int alreadyComplained = 0;
    
    if(alreadyComplained == 0)
    {
        alreadyComplained = 1;
        
        switch(err)
        {
            case MuBuErrorNone:
                break;
                
            case MuBuErrorNoLibrary:
            {
                error("%s: cannot find MuBu container", class_nameget(c)->s_name);
                return 0;
            }
                
                break;
                
            default:
            {
                static int alreadyComplained = 0;
                
                if(alreadyComplained == 0)
                {
                    error("%s: client interface version mismatches loaded MuBu container", class_nameget(c)->s_name);
                    return 0;
                }
            }
        }
    }
    
    return 1;
}

#pragma mark -
#pragma mark ...


MubuTrainingSet_class_init(t_class *cl)
{
    if(c != NULL)
    {
        class_addmethod(c, (method)notifyMethod, "notify", A_CANT, 0);
        class_addmethod(c, (method)referMethod,  "refer", A_GIMME, 0);
    }
}

static t_max_err notifyMethod(MubuTrainingSet *self, t_symbol *s, t_symbol *msg, void *sender, void *data)
{
    systhread_mutex_lock(self->mutex);
    
    /* handle change of mubu/imubu bound to name */
    if(msg == gensym("globalsymbol_binding"))
        mubuRef_connect(&self->container, (t_object *)self, self->containerName);
    else if(msg == gensym("globalsymbol_unbinding"))
        mubuRef_connect(&self->container, (t_object *)self, NULL);
    
    if(self->autoUpdate &&sender == self->container)
    {
        qelem_front(self->updateQelem);
    }
    
    systhread_mutex_unlock(self->mutex);
    
    return MAX_ERR_NONE;
}

static void referMethod(MubuTrainingSet *self, t_symbol *s, short ac, t_atom *at)
{
    if(ac > 0 && atom_issym(at))
        self->trainingSet->setContainerName(atom_getsym(at));
}

void updateTrainingSet(MubuTrainingSet *self)
{
    self->trainingSet->updateTrainingSet();
}


#endif
