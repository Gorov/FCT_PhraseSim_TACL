//
//  BaseLearner.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-4-25.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_BaseLearner_h
#define PhraseEmb_BaseLearner_h


#include "FeatureModel.h"
#include "EmbeddingModel.h"
#include <math.h>
#include <stdlib.h>

#define MAX_EXP 86
#define MIN_EXP -86

class Options
{
    public:
    bool halflm;
    bool unnorm;
};

class BaseLearner
{
public:
    string embfile;
    //string clusfile;
    
    EmbeddingModel* emb_model;
    FeatureModel* fea_model;
    BaseInstance* inst;
    
    word2int train_word_vocab;
    word2int train_phrase_vocab;
    word2int train_word_out_vocab;
    
    /*
    double* lambda1;
    double* lambda2;
    double* emb_p;
    
    double* part_lambda1;
    double* part_lambda2;
    double* part_emb_p;
    */
    real* lambda1;
    real* lambda2;
    real* emb_p;
    
    real* part_lambda1;
    real* part_lambda2;
    real* part_emb_p;
    
    int feat_vec1[256];
    int feat_vec2[256];
    int num_feat1;
    int num_feat2;
    long long layer1_size;
    
    /*
    double eta0;
    double eta;
    */
    real eta0;
    real eta;
    int iter;
    unsigned long num_fea;
    bool update_emb;
    bool five_choice;
    
    BaseLearner() {};
    
    BaseLearner(char* embfile) {
        //Init();
        this -> embfile = embfile;
    }
    ~BaseLearner(){
        delete emb_model;
        delete fea_model;
        
        delete[] lambda1;
        delete[] lambda2;
        delete [] emb_p;
        delete [] part_lambda1;
        delete [] part_lambda2;
        delete [] part_emb_p;
    }
    
    BaseLearner(char* embfile, char* trainfile){
        Init(embfile, trainfile);
        this -> embfile = embfile;
    }
    BaseLearner(char* embfile, char* clusfile, char* trainfile){
        Init(embfile, clusfile, trainfile);
        this -> embfile = embfile;
    }
    BaseLearner(char* embfile, char* clusfile, char* trainfile, int type){
        Init(embfile, clusfile, trainfile, type);
        this -> embfile = embfile;
    }
    
    BaseLearner(char* freqfile, char* clusfile, char* trainfile, bool random_emb){
        Init(freqfile, clusfile, trainfile, random_emb);
        this -> embfile = embfile;
    }
    
    void TrainData(string trainfile);
    void TrainData(string trainfile, string trainsubfile, string devfile);
    virtual void EvalData(string trainfile) = 0;
    virtual int LoadInstance(ifstream& ifs) = 0;
    
    void ExtractFeature();
    void ComputeLambdas();
    virtual void ForwardOutputs() = 0;
    void BackPropFeatures();
    //virtual void BackPropEmb();
    virtual long BackPropPhrase() = 0;
    
    void BuildVocab(string trainfile);
    
    void SaveModel(string modelfile, bool savelm);
    void SaveModel(string modelfile, bool savelm, int type);
    void SaveModel(string modelfile);
    void LoadModel(string modelfile);
    
    void ForwardProp();
    void BackProp();
    
    void GetGradients();
    
    void Init();
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata);
    void Init(char* embfile, char* clusfile, char* traindata, int type);
    void Init(char* freqfile, char* clusfile, char* traindata, bool random_emb);
    
    void Delete();
};


#endif
