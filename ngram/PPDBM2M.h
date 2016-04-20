//
//  PPDBM2M.h
//  Preposition_Classification
//
//  Created by gflfof gflfof on 15-1-14.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef Preposition_Classification_PPDBM2M_h
#define Preposition_Classification_PPDBM2M_h

#include "PPDBInstance.h"
#include "FctCoarseModel.h"
#include "FctDeepModel.h"
#include "FctConvolutionModel.h"
#include "FeatureModel.h"

#define THRES1 1000
#define THRES2 10000

class PPDB_Params {
public:
    bool sum;
    bool position;
    bool head;
    bool postag;
    bool clus;
    
    bool low_rank;
    
    void PrintValue() {
        cout << "sum:" << sum << endl;
        cout << "position:" << position << endl;
        cout << "head:" << head << endl;
        cout << "postag:" << postag << endl;
        cout << "clus:" << clus << endl;
    }
};

class PPDBM2M
{
public:
    string type;
    bool adagrad;
    bool update_emb;
    
    vector<FctDeepModel*> deep_fct_list;
    vector<EmbeddingModel*> emb_model_list;
    EmbeddingModel* emb_model;
    
    int num_models;
    int layer1_size;
    int num_inst;
    int max_len;
    
    feat2int slot2deep_model;
    vector<string> deep_slot_list;
    
    
    int num_labels;
    BaseInstance* b_inst;
    PPDBNgramInstance* inst;
    PPDB_Params fea_params;
    
    real* label_scores;
    real* label_embeddings;
    
    real eta0;
    real eta;
    real eta_real;
    real alpha_old;
    real alpha;
    real lambda;
    real lambda_prox;
    
    int iter;
    int cur_iter;
    
    //for NCE
    word2int phrase_vocab;
    vector<string> vocab;
    unsigned long long next_random;
    const int table_size = 1e8;
    int *table;
    long vocab_size;
    int* freqtable;
    word_info* wordinfo_table;
    
    PPDBM2M() {b_inst = new BaseInstance(); inst = new PPDBNgramInstance();}
    ~PPDBM2M() {}
    
    PPDBM2M(char* embfile, char* trainfile) {
        type = "PPDBM2M";
        b_inst = new BaseInstance();
        inst = new PPDBNgramInstance();
        Init(embfile, trainfile);
    }
    
    void BuildModelsFromData(char* trainfile);
    void InitSubmodels();
    void InitVocab(char* vocabfile);
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    void RandomInstance();
    void GetInstanceById(int pos);
    int SearchDeepFctSlot(string slot_key);
    int AddDeepFctModel(string slot_key);
    
    int SearchFeature(string feat_key);
    int AddFeature(string feat_key);
    
    string ToLower(string& s);
    
    void Init(char* embfile, char* traindata);
    
    void ForwardProp();
    void ForwardNCE(ifstream& ifs);
    void BackProp();
    
    virtual void TrainData(string trainfile, string devfile, int type);
//    virtual void EvalData(string trainfile, int type);
    void EvalLogLoss(string testfile);
    void EvalMRR(string testfile, int threshold);
    //    virtual void EvalData(string trainfile, string outfile, int type);
    
    void PrintModelInfo();
    void WeightDecay(real eta_real, real lambda);
    
    //    void PushWordFeature(string slot_key);
    void PushWordFeature(string slot_key, int wordid, FctDeepModel* p_model, RealFctPathInstance* p_inst);
    
    void ExtractFeatures();
    
    void InitFreqTable(char* filename);
    void InitUnigramTable();
    long SampleNegative();
};

#endif
