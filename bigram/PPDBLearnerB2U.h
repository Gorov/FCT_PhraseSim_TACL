//
//  PPDBLearnerB2U.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-15.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_PPDBLearnerB2U_h
#define PhraseEmb_PPDBLearnerB2U_h

#include "BaseLearner.h"
#include "Commons.h"

class PPDBLearnerB2U: public BaseLearner
{
public:    
    vector<string> vocab;
    unsigned long long next_random;
    const int table_size = 1e8;
    int *table;
    long vocab_size;
    int* freqtable;
    word_info* wordinfo_table;
    
    //bool halflm;
    //bool unnorm;
    
    PPDBLearnerB2U();
    ~PPDBLearnerB2U() {}
    
    PPDBLearnerB2U(char* embfile):
    BaseLearner(embfile) {
        Init();
    }
    
    PPDBLearnerB2U(char* embfile, char* trainfile):
    BaseLearner(embfile, trainfile){
        Init(embfile, trainfile);
    }
    PPDBLearnerB2U(char* embfile, char* clusfile, char* trainfile):
    BaseLearner(embfile, clusfile, trainfile, EMB){
    //BaseLearner(embfile, clusfile, trainfile){
        Init(embfile, clusfile, trainfile);
    }
    
    PPDBLearnerB2U(char* embfile, char* clusfile, char* trainfile, int type):
    BaseLearner(embfile, clusfile, trainfile, type){
        //BaseLearner(embfile, clusfile, trainfile){
        Init(embfile, clusfile, trainfile);
    }
    
    void BuildBinaryTree();
    
    void TrainBigData(string trainfile, string trainsubfile, string devfile);
    
    void InitFreqTable(char* filename);
    void InitUnigramTable();
    long SampleNegative();
    
    void EvalData(string trainfile);
    void EvalLogLoss(string trainfile);
    void EvalMRR(string testfile, int threshold);
    void EvalMRRDouble(string testfile, int threshold);
    
    void BuildVocab(string trainfile);
    void EvalDataObs(string trainfile);
    void EvalMRRObs(string trainfile, int threshold);
    
    int LoadInstance(ifstream& ifs);
    void ForwardOutputs();
    long BackPropPhrase();
    
    void LoadModel(string modelfile);
    
    void LoadModel(string modelfile, string embfile);
    void LoadEmb(string embfile);
    
    void Init();
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata);
    void InitWithOption(char* embfile, char* clusfile, char* trainfile);
    void InitVocab(char* vocabfile);
};

#endif
