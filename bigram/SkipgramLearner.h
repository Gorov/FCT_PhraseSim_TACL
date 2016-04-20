//
//  SkipgramLearner.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-2.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_SkipgramLearner_h
#define PhraseEmb_SkipgramLearner_h

#include "BaseLearner.h"
#include "Commons.h"

class SkipgramLearner: public BaseLearner
{
public:    
    vector<string> vocab;
    unsigned long long next_random;
    const int table_size = 1e8;
    int *table;
    long vocab_size;
    int* freqtable;
    word_info* wordinfo_table;
    
    const int exp_table_size = 1000;
    const int max_exp = 6;
    real *expTable;
    
    bool hs;
    real scores[10 * 40];
    
    bool negative;
    
    bool halflm;
    bool unnorm;
    
    SkipgramLearner() {};
    ~SkipgramLearner() {}
    
    SkipgramLearner(char* embfile):
    BaseLearner(embfile) {
        Init();
    }
    
    SkipgramLearner(char* embfile, char* trainfile):
    BaseLearner(embfile, trainfile){
        Init(embfile, trainfile);
    }
    SkipgramLearner(char* embfile, char* clusfile, char* trainfile):
    BaseLearner(embfile, clusfile, trainfile){
        Init(embfile, clusfile, trainfile);
    }
    SkipgramLearner(char* embfile, char* clusfile, char* trainfile, int type):
    BaseLearner(embfile, clusfile, trainfile, type){
        Init(embfile, clusfile, trainfile);
    }
    
    SkipgramLearner(char* freqfile, char* clusfile, char* trainfile, bool random_emb):
    BaseLearner(freqfile, clusfile, trainfile, random_emb){
        Init(freqfile, clusfile, trainfile);
    }
    
    void BuildBinaryTree();
    void InitExpTable();
    
    void TrainBigData(string trainfile, string trainsubfile, string devfile);
    
    void InitFreqTable(char* filename);
    void InitUnigramTable();
    long SampleNegative();
    
    void EvalData(string trainfile);
    void EvalLogLoss(string trainfile);
    void EvalLogLossBase(string trainfile);
    
    int LoadInstance(ifstream& ifs);
    void ForwardOutputs();
    long BackPropPhrase();
    void ForwardBaselineLeft();
    void ForwardBaselineRight();
    
    //void Savemodel(string modelfile);
    void LoadModel(string modelfile, bool loadlm);
    
    void Init();
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata);
    void InitWithOption(char* embfile, char* clusfile, char* trainfile);
    void InitVocab(char* vocabfile);
};

#endif
