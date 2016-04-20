//
//  BinaryLearnerNew.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-2.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_BinaryLearnerNew_h
#define PhraseEmb_BinaryLearnerNew_h

#include "BaseLearner.h"

class BinaryLearnerNew: public BaseLearner
{
public:    
    
    BinaryLearnerNew();
    ~BinaryLearnerNew() {}
    
    BinaryLearnerNew(char* embfile, char* trainfile):
    BaseLearner(embfile, trainfile){
        Init(embfile, trainfile);
    }
    BinaryLearnerNew(char* embfile, char* clusfile, char* trainfile):
    BaseLearner(embfile, clusfile, trainfile){
        Init(embfile, clusfile, trainfile);
    }
    
    void EvalData(string trainfile);
    
    int LoadInstance(ifstream& ifs);
    void ForwardOutputs();
    long BackPropPhrase();
    
    double Sigmoid(double score);
    
    //void Savemodel(string modelfile);
    
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata);
};

#endif
