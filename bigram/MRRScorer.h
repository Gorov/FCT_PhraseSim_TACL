//
//  MRRScorer.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-14.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_MRRScorer_h
#define PhraseEmb_MRRScorer_h

#include "BaseLearner.h"

class MRRScorer: public BaseLearner
{
public:    
    word2int phrasedict;
    real* phraseemb;
    
    PPDBTargetInstance* target_inst;
    
    bool unnorm;
    
    MRRScorer();
    ~MRRScorer() {}
    
    MRRScorer(char* embfile):
    BaseLearner(embfile) {
        Init();
    }
    
    MRRScorer(char* embfile, char* trainfile):
    BaseLearner(embfile, trainfile){
        Init(embfile, trainfile);
    }
    MRRScorer(char* embfile, char* clusfile, char* trainfile):
    BaseLearner(embfile, clusfile, trainfile){
        Init(embfile, clusfile, trainfile);
    }
    
    void EvalData(string trainfile) {};
    void EvalLogLoss(string trainfile) {};
    void EvalMRR(string testfile, int threshold, string phrase_list, bool withphrase);
    
    int LoadInstance(ifstream& ifs);
    
    void PrecomputeEmb(word2int& phrasedict, real* phraseemb);
    void ForwardProp();
    virtual void ForwardOutputs() {};
    virtual long BackPropPhrase() {};
    void LoadModel(string modelfile);
    void LoadModel(string modelfile, bool onlyemb);
    
    void Init();
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata);
};

#endif
