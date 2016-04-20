//
//  MTLTrainer.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-24.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_MTLTrainer_h
#define PhraseEmb_MTLTrainer_h

#include "BaseLearner.h"

//#define BINARY 0
//#define PPDB 1
//#define SKIPGRAM 2
#define BINARY_INST_DS 5
#define SKIPGRAM_INST_DS 6

class MTLTrainer: public BaseLearner
{
public:    
    double b;
    BinaryInstance* b_inst;
    PPDBB2UInstance* p_inst;
    SkipgramInstance* s_inst;
    
    int b_inst_num;
    int p_inst_num;
    int s_inst_num;
    
    real C;
    
    real* syn0;
    
    real inner_norm;
    real norm1;
    real* e2_norm;
    real norm2;
    
    real* scores;
    real* norms;
    real* e_norms;
    
    vector<string> vocab;
    unsigned long long next_random;
    const int table_size = 1e8;
    int *table;
    long vocab_size;
    int* freqtable;
    word_info* wordinfo_table;
    
    word2int maintask_vocab;
    bool mask;
    
    MTLTrainer();
    ~MTLTrainer() {}
    
    MTLTrainer(char* embfile, char* clusfile, char* traindata_b, char* traindata_p, char* traindata_s, int type):
    BaseLearner(embfile, clusfile, traindata_b, type){
        Init(embfile, clusfile, traindata_b, traindata_p, traindata_s);
    }
    
    void EvalData(string testfile, int obj_type);
    void EvalMRR(string testfile, int threshold);
    void EvalLogLoss(string testfile);
    
    int LoadInstance(ifstream& ifs, int obj_type);
    void LoadMaintaskVocab(string trainfile, int type);
    void ForwardOutputs(int obj_type);
    long BackPropPhrase(int obj_type);
    
    void ForwardProp(int obj_type);
    void BackProp(int obj_type);
    
    void BackPropFeatures(real C);
    
    double Sigmoid(double score);
    
    //void Savemodel(string modelfile);
    
    void Init(char* embfile, char* clusfile, char* traindata_b, char* traindata_p, char* traindata_s);
    void TrainData(char* traindata_b, char* traindata_p, char* traindata_s);
    void TrainDataNew(char* traindata_b, char* traindata_s);
    void TrainDataNew(char* traindata_s);
    void TrainBigData(char* traindata_b, char* traindata_s, char* dev_file_b);
    void TrainBigDataNew(char* traindata_b, char* traindata_s, char* dev_file_b, char* test_file_b, int type);
    
    void InitVocab(char* vocabfile);
    void InitFreqTable(char* filename);
    void InitUnigramTable();
    long SampleNegative();
    
    //no use
    void EvalData(string trainfile) {};
    int LoadInstance(ifstream& ifs) {return 0;}
    void ForwardOutputs() {};
    long BackPropPhrase() {return 0;}
};


#endif
