//
//  EmbeddingModel.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-4-14.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_EmbeddingModel_h
#define PhraseEmb_EmbeddingModel_h

#include <tr1/unordered_map>
#include <iostream>
#include <vector>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "Commons.h"

#define EMB 0
#define HALFLM 1
#define LM 2
#define NORMEMB 3
#define HSLM 4
#define RANDEMB 10

using namespace std;

typedef std::tr1::unordered_map<string, int> word2int;
typedef float real;

class EmbeddingModel
{
public:
    word2int vocabdict;
    vector<string> vocablist;
    long vocab_size;
    long long layer1_size;
    real *syn0;
    real *syn1neg;
    real *syn1;
    
    vector<int> freqlist;
    
    EmbeddingModel() {};
    
    EmbeddingModel(char* modelname, int type) {
        if (type == EMB) LoadEmbUnnorm(modelname);
        else if (type == HALFLM) LoadHalfLM(modelname);
        else if (type == LM) LoadLM(modelname);
        else if (type == NORMEMB) LoadEmb(modelname);
        else if (type == RANDEMB) InitEmb(modelname, 200);
        else if (type == HSLM) LoadHSLM(modelname);
        vocab_size = vocabdict.size();
    }
    
    EmbeddingModel(char* modelname) {
        //LoadEmb(modelname);
        LoadHalfLM(modelname);
        //LoadLM(modelname);
        vocab_size = vocabdict.size();
    }
    
    EmbeddingModel(char* modelname, bool unnorm, bool halflm) {
        //LoadEmb(modelname);
        if (halflm) LoadHalfLM(modelname);
        else LoadLM(modelname);
        vocab_size = vocabdict.size();
    }
    
    EmbeddingModel(char* modelname, bool unnorm) {
        if (unnorm) LoadEmbUnnorm(modelname);
        else LoadEmb(modelname);
        vocab_size = vocabdict.size();
    }
    
    int LoadEmb(string modelname);
    
    int LoadEmbUnnorm(string modelname);
    int LoadLM(string modelname);
    int LoadHSLM(string modelname);
    int LoadHalfLM(string modelname);
    
    int LoadLMAdditively(string modelname);
    
    int InitEmb(char* freqfile, int dim);
    
    void SaveEmb(string modelfile);
    void SaveLM(string modelfile);
    void SaveLM(string modelfile, int type);
    
    //for joint:
    void InitFromTrainFile(char* train_file);
    void LearnVocabFromTrainFile(char* train_file);
    void InitNet();
    void InitNetHS();
    void SortVocab();
};

#endif
