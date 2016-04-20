//
//  PPDBLearner.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-8.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_PPDBLearner_h
#define PhraseEmb_PPDBLearner_h

#include "BaseLearner.h"
#include "Commons.h"

#define NUM_NEG_INST 1

class PPDBLearner: public BaseLearner
{
public:    
    vector<string> vocab;
    unsigned long long next_random;
    const int table_size = 1e8;
    int *table;
    long vocab_size;
    int* freqtable;
    word_info* wordinfo_table;
    
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
    PPDBTargetInstance* neg_inst[NUM_NEG_INST];
    
    real* neg_target_lambda1[NUM_NEG_INST];
    real* neg_target_lambda2[NUM_NEG_INST];
    real* neg_target_emb_p[NUM_NEG_INST];
    
    real* neg_part_target_lambda1[NUM_NEG_INST];
    real* neg_part_target_lambda2[NUM_NEG_INST];
    real* neg_part_target_emb_p[NUM_NEG_INST];
    
    real neg_target_scores[NUM_NEG_INST];
    
    int neg_target_feat_vec1[NUM_NEG_INST][256];
    int neg_target_feat_vec2[NUM_NEG_INST][256];
    int neg_target_num_feat1[NUM_NEG_INST];
    int neg_target_num_feat2[NUM_NEG_INST];
    
    bool unnorm;
    
    PPDBLearner();
    ~PPDBLearner() {}
    
    PPDBLearner(char* embfile):
    BaseLearner(embfile) {
        Init();
    }
    
    PPDBLearner(char* embfile, char* trainfile):
    BaseLearner(embfile, trainfile){
        Init(embfile, trainfile);
    }
    PPDBLearner(char* embfile, char* clusfile, char* trainfile):
    BaseLearner(embfile, clusfile, trainfile, EMB){
        Init(embfile, clusfile, trainfile);
    }
    
    void TrainBigData(string trainfile, string trainsubfile, string devfile);
    
    void InitFreqTable(char* filename);
    void InitUnigramTable();
    long SampleNegative();
    
    void EvalData(string trainfile);
    void EvalLogLoss(string trainfile);
    //void EvalLogLossBase(string trainfile);
    
    int LoadInstance(ifstream& ifs);
    
    void GetNegInsts();
    void ComposeTarget();
    void ExtractFeatureTarget();
    void ComputeLambdasTarget();
    
    void BackPropFeaturesTarget();
    void BackPropTarget();
    
    void ForwardOutputs();
    long BackPropPhrase();
//    void ForwardBaselineLeft();
//    void ForwardBaselineRight();
    
    //void Savemodel(string modelfile);
    void LoadModel(string modelfile);
    
    void Init();
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata);
    void InitArray();
    //void InitWithOption(char* embfile, char* clusfile, char* trainfile);
    void InitVocab(char* vocabfile);
};

#endif
