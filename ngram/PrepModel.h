//
//  PrepModel.h
//  Preposition_Classification
//
//  Created by gflfof gflfof on 14-12-24.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef Preposition_Classification_PrepModel_h
#define Preposition_Classification_PrepModel_h

#include "Instances.h"
#include "FctCoarseModel.h"
#include "FctDeepModel.h"
#include "FctConvolutionModel.h"
#include "FeatureModel.h"

class Prep_Params {
public:
    bool sum;
    bool position;
    bool prep;
    bool clus;
    
    bool low_rank;
    
    bool tri_conv;
    bool linear;
    
    void PrintValue() {
        cout << "position:" << position << endl;
        cout << "prep:" << prep << endl;
        
        cout << "tri_conv:" << tri_conv << endl; 
        cout << "linear:" << linear << endl; 
    }
};

class PrepModel
{
public:
    string type;
    bool adagrad;
    bool update_emb;
    
    vector<FctCoarseModel*> coarse_fct_list;
    vector<FctDeepModel*> deep_fct_list;
    vector<FctConvolutionModel*> convolution_fct_list;
    vector<EmbeddingModel*> emb_model_list;
    EmbeddingModel* emb_model;
    
    int num_models;
    int layer1_size;
    int num_inst;
    int max_len;
    
    feat2int slot2coarse_model;
    vector<string> coarse_slot_list;
    feat2int slot2deep_model;
    vector<string> deep_slot_list;
    
    feat2int slot2convolution_model;
    vector<string> convolution_slot_list;
    
    feat2int labeldict;
    vector<string> labellist;
    
    int num_labels;
//    BaseInstance* inst;
    PrepInstance* inst;
    Prep_Params fea_params;
    
    real eta0;
    real eta;
    real eta_real;
    real alpha_old;
    real alpha;
    real lambda;
    real lambda_prox;
    
    int iter;
    int cur_iter;
    
    PrepModel() {inst = new PrepInstance();}
    ~PrepModel() {}
    
    PrepModel(char* embfile, char* trainfile) {
        type = "PREP";
        inst = new PrepInstance();
        Init(embfile, trainfile);
    }
    
    void BuildModelsFromData(char* trainfile);
    void InitSubmodels();
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    int LoadInstance(ifstream& ifs, int type);
    int LoadInstanceSemEval(ifstream& ifs);
    int SearchCoarseFctSlot(string slot_key);
    int AddCoarseFctModel(string slot_key);
    int SearchDeepFctSlot(string slot_key);
    int AddDeepFctModel(string slot_key);
    
    int SearchFeature(string feat_key);
    int AddFeature(string feat_key);
    
    int SearchConvolutionSlot(string slot_key);
    int AddConvolutionModel(string slot_key, int length);
    
    int AddCoarseFctModel2List(string slot_key, int count, bool add);
    int AddConvolutionModel2List(string slot_key, vector<int> word_id_vec, bool add);
    
    string ProcSenseTag(string input_type);
    string ProcNeTag(string input_type);
    string ToLower(string& s);
    
    void Init(char* embfile, char* traindata);
    
    void ForwardProp();
    void BackProp();
    
    virtual void TrainData(string trainfile, string devfile, int type);
    virtual void EvalData(string trainfile, int type);
//    virtual void EvalData(string trainfile, string outfile, int type);
    
    void PrintModelInfo();
    void WeightDecay(real eta_real, real lambda);
    
//    void PushWordFeature(string slot_key);
    void PushWordFeature(string slot_key, int wordid, FctDeepModel* p_model, RealFctPathInstance* p_inst);
};


#endif
