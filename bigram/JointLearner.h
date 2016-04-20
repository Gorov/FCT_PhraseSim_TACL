//
//  JointLearner.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-17.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_JointLearner_h
#define PhraseEmb_JointLearner_h

#include "BaseLearner.h"
#include "SkipgramLearner.h"
#include "Commons.h"
#include <pthread.h>

class ThreadPara {
    public:
    void* pointer; 
    long thread_id;
    ThreadPara(void* p, long i) {
        pointer = p;
        thread_id = i;
    }
};

class ThreadPara2 {
public:
    void* pointer; 
    string filename;
    ThreadPara2(void* p, string f) {
        pointer = p;
        filename = f;
    }
};

class JointLearner: public SkipgramLearner
{
public:    
    int num_threads;
    int window;
    int negative;
    real alpha0;
    real lm_alpha;
    
    bool init_vocab;
    
    real sample;
    
    string train_lm_file;
    long long train_words;
    long long word_count_actual;
    real *expTable;
    
    JointLearner() {};
    
    ~JointLearner() {}
    
    JointLearner(char* embfile):
    SkipgramLearner(embfile) {
        //Init();
    }
    
    JointLearner(char* embfile, char* trainfile):
    SkipgramLearner(embfile, trainfile){
        //Init(embfile, trainfile);
    }
    JointLearner(char* embfile, char* clusfile, char* trainfile):
    SkipgramLearner(embfile, clusfile, trainfile){
        //Init(embfile, clusfile, trainfile);
    }
    
    JointLearner(char* freqfile, char* clusfile, char* trainfile, bool random):
    SkipgramLearner(freqfile, clusfile, trainfile,random){
    }
    
    JointLearner(char* freqfile, char* clusfile, char* trainfile, int type):
    SkipgramLearner(freqfile, clusfile, trainfile, type){
    }
    
    void DefaultInit() {
        lm_alpha = alpha0 = 0.025;
        num_threads = 12;
        window = 5;
        negative = 15;
        sample = 1e-3;
    }
    
    void InitTrainer(real alpha, real sample, int window, int negative, string trainfile, int threads) {
        lm_alpha = alpha0 = alpha;
        this -> sample = sample;
        this -> window = window;
        this -> negative = negative;
        train_lm_file = trainfile;
        num_threads = threads;
    }
    
    void InitExpTable();
    
    void InitTrainInfoLM();
    void InitTrainInfoLM(const char* filename);
    
    void InitVocab();
    
    void *TrainWordEmbThread(long thread_id);
    void ReadWord(char *word, FILE *fin);
    int ReadWordIndex(FILE *fin);
    
    //void TrainBigData(string trainfile, string trainsubfile, string devfile);
    void JointTrainBigData(string trainfile, string trainfile_lm, string trainsubfile, string devfile);
    
    static void* threadFunction(void *obj) {
        reinterpret_cast< JointLearner *> (((ThreadPara*)obj) -> pointer)->TrainWordEmbThread(((ThreadPara*)obj)->thread_id);
        return NULL;
    }
    static void* threadFunction2(void *obj) {
        reinterpret_cast< JointLearner *> (((ThreadPara2*)obj) -> pointer)->TrainData(((ThreadPara2*)obj) -> filename);
        return NULL;
    }
    
    void ForwardOutputs();
    long BackPropPhrase();
    
    //void Savemodel(string modelfile);
    void LoadModel(string modelfile);
    
    //void InitVocab(char* vocabfile);
};


#endif
