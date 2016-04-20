//
//  MultiBinaryLearner.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-2.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_MultiBinaryLearner_h
#define PhraseEmb_MultiBinaryLearner_h

#include "BaseLearner.h"

class MultiBinaryLearner: public BaseLearner
{
public:    
    
    MultiBinaryLearner();
    ~MultiBinaryLearner() {}
    
    MultiBinaryLearner(char* embfile, char* trainfile):
    BaseLearner(embfile, trainfile){
        Init(embfile, trainfile);
    }
    MultiBinaryLearner(char* embfile, char* clusfile, char* trainfile):
    BaseLearner(embfile, clusfile, trainfile){
        Init(embfile, clusfile, trainfile);
    }
    
    
    void EvalData14(string trainfile);
    void EvalData(string trainfile);
    void EvalDataAcc(string trainfile);
    
    int LoadInstance(ifstream& ifs);
    void ForwardOutputs();
    long BackPropPhrase();
    
    double Sigmoid(double score);
    
    //void Savemodel(string modelfile);
    
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata);
};

#endif
