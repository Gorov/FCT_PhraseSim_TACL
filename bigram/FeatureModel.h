//
//  FeatureModel.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-4-14.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_FeatureModel_h
#define PhraseEmb_FeatureModel_h

#include <tr1/unordered_map>
#include <iostream>
#include <stdlib.h>
#include "Instance.h"
#include "EmbeddingModel.h"

#define BINARY_CLASS 0
#define MULTI_CLASS 1
#define NCE 2

//#define REG_INST 0
//#define SKIPGRAM_INST 2
//#define PPDB_INST 3

#define BINARY_INST 0
#define PPDB_INST 1
#define SKIPGRAM_INST 2

using namespace std;

typedef std::tr1::unordered_map<string, int> feat2int;
typedef std::tr1::unordered_map<string, string> word2clus;

class FeatureModel
{
    public:
    int dim;
    unsigned long num_fea;
    unsigned long num_inst;
    feat2int feadict;
    word2clus clusdict;
    /*
    double* param;
    double b1;
    double b2;
    double* b1s;
    double* b2s;
    */
    real* param;
    real* b1s;
    real* b2s;
    bool lex_fea;
    FeatureModel() {};
    FeatureModel(int dim, char* filename){
        this -> dim = dim;
        InitFeatDict(filename);
        lex_fea = false;
    }
    FeatureModel(int dim, char* filename, int type){
        this -> dim = dim;
        if (type == 0) {
            InitFeatDictBinary(filename);
        }
        else if (type == 1) {
            InitFeatDict(filename);
        }
        else if (type == 2) {
            InitFeatDictSkipgram(filename);
        }
        lex_fea = false;
    }
    FeatureModel(int dim, char* filename, char* clusfile){
        this -> dim = dim;
        lex_fea = true;
        InitClusDict(clusfile);
        InitFeatDict(filename);
    }
    FeatureModel(int dim, char* filename, char* clusfile, int type){
        this -> dim = dim;
        lex_fea = true;
        InitClusDict(clusfile);
        if (type == 0) {
            InitFeatDictBinary(filename);
        }
        else if (type == 1) {
            InitFeatDict(filename);
        }
        else if (type == 2) {
            InitFeatDictSkipgram(filename);
        }
        else if (type == PPDB_INST) {
            InitFeatDict(filename, type);
        }
        //InitFeatDict(filename);
        lex_fea = true;
    }
    void ExtractFeaBNP(Instance *inst, int* feat_vec1, int* feat_vec2, int* counts);
    //void CollectFeaBNP(Instance *inst);
    
    void ExtractFeaBNP(BaseInstance *inst, int* feat_vec1, int* feat_vec2, int* counts);
    void CollectFeaBNP(BaseInstance *inst);
    void InitFeatDict(char* filename);
    void InitFeatDictBinary(char *filename);
    void InitFeatDictSkipgram(char *filename);
    void InitFeatDict(char* filename, int type);
    void InitClusDict(char* filename);
    int LoadInstance(ifstream &ifs, Instance* inst);
    int LoadInstance(ifstream &ifs, BinaryInstance* inst);
    int LoadInstance(ifstream &ifs, SkipgramInstance* inst);
    int LoadInstance(ifstream &ifs, PPDBInstance* inst);
    
    void InitFeatPara();
    void InitAddtionalFeatPara();
    int AddFeatDict(char *filename, int type);
    
    void SaveModel(string modelfile);
    void LoadModel(string modelfile);
};

#endif
