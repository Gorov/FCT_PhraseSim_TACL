//
//  GetPhraseSim.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-9.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_GetPhraseSim_h
#define PhraseEmb_GetPhraseSim_h

#include "BaseLearner.h"

class GetPhraseSim: public BaseLearner
{
public:    
    vector<string> vocab;
    unsigned long long next_random;
    const int table_size = 1e8;
    int *table;
    long vocab_size;
    int* freqtable;
    
    real* target_lambda1;
    real* target_lambda2;
    real* target_emb_p;
    
    real* part_target_lambda1;
    real* part_target_lambda2;
    real* part_target_emb_p;
    
    int target_feat_vec1[256];
    int target_feat_vec2[256];
    int target_num_feat1;
    int target_num_feat2;
    
    PPDBTargetInstance* target_inst;
    
    bool unnorm;
    
    GetPhraseSim();
    ~GetPhraseSim() {}
    
    GetPhraseSim(char* embfile):
    BaseLearner(embfile) {
        //LoadEmb(embfile, false);
        //Init();
    }
    
    GetPhraseSim(char* embfile, char* trainfile):
    BaseLearner(embfile, trainfile){
        Init(embfile, trainfile);
    }
    GetPhraseSim(char* embfile, char* clusfile, char* trainfile):
    BaseLearner(embfile, clusfile, trainfile){
        Init(embfile, clusfile, trainfile);
    }
    
    void GetSim(string testfile, string outfile);
    //void EvalLogLossBase(string trainfile);
    
    void EvalData(string trainfile) {}
    int LoadInstance(ifstream& ifs);
    
    void ComposeTarget();
    void ExtractFeatureTarget();
    void ComputeLambdasTarget();
    
    void ForwardOutputs() {}
    long BackPropPhrase() {}

    void LoadModel(string modelfile, bool loadlm);
    void LoadEmb(string modelfile, bool unnorm);
    
    void Init();
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata);
    void InitArray();
    //void InitWithOption(char* embfile, char* clusfile, char* trainfile);
    void InitVocab(char* vocabfile);

};


#endif
