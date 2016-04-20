//
//  ClassificationLearnerNew.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-1.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_ClassificationLearnerNew_h
#define PhraseEmb_ClassificationLearnerNew_h

#include "BaseLearner.h"

class ClassificationLearnerNew: public BaseLearner
{
public:    
    
    ClassificationLearnerNew() {inst = new Instance();};
    ~ClassificationLearnerNew() {}
    
    ClassificationLearnerNew(char* embfile, char* trainfile):
    BaseLearner(embfile, trainfile){
        Init(embfile, trainfile);
    }
    ClassificationLearnerNew(char* embfile, char* clusfile, char* trainfile):
    BaseLearner(embfile, clusfile, trainfile){
        Init(embfile, clusfile, trainfile);
    }
    
    
    void EvalData14(string trainfile);
    void EvalData(string trainfile);
    void EvalData14DS(string trainfile);
    void EvalDataDS(string trainfile);
    
    int LoadInstance(ifstream& ifs);
    void ForwardPropDS();
    void ForwardOutputsDS();
    void ForwardOutputs();
    long BackPropPhrase();
    
    //void Savemodel(string modelfile);
    
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata);
    
    void LoadModel(string modelfile, string feafile, int type);
    void LoadModel(string modelfile, string feafile, string embfile);
};

#endif
