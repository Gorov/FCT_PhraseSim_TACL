//
//  BinaryThresLearner.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-4.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_BinaryThresLearner_h
#define PhraseEmb_BinaryThresLearner_h

#include "BaseLearner.h"

class BinaryThresLearner: public BaseLearner
{
public:    
    double b;
    
    BinaryThresLearner();
    ~BinaryThresLearner() {}
    
    BinaryThresLearner(char* embfile, char* trainfile):
    BaseLearner(embfile, trainfile){
        Init(embfile, trainfile);
    }
    BinaryThresLearner(char* embfile, char* clusfile, char* trainfile):
    BaseLearner(embfile, clusfile, trainfile, HALFLM){
        Init(embfile, clusfile, trainfile);
    }
    
    void EvalData(string trainfile);
    
    void BuildVocab(string trainfile);
    void EvalDataObs(string trainfile);
    
    int LoadInstance(ifstream& ifs);
    void ForwardOutputs();
    long BackPropPhrase();
    
    double Sigmoid(double score);
    
    //void Savemodel(string modelfile);
    
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata);
};

#endif
