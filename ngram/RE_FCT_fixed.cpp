//
//  RE_FCT.cpp
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-31.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sstream>
#include <limits>
#include "FullFctModel.h"
#include "PrepModel.h"
#include "PPDBM2M.h"

#define SEM_EVAL 1
#define ACE_2005 2
#define PREP 3

char train_file[MAX_STRING], dev_file[MAX_STRING], res_file[MAX_STRING];
char output_file[MAX_STRING], param_file[MAX_STRING];
char clus_file[MAX_STRING], baseemb_file[MAX_STRING], freq_file[MAX_STRING];
char model_file[MAX_STRING];
char feature_file[MAX_STRING];
char trainsub_file[MAX_STRING];
int iter = 1;
int finetuning = 0;
real alpha = 0.01;
real eta0 = 0.05;

int ArgPos(char *str, int argc, char **argv);

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    output_file[0] = 0;
//    string dir = "/Users/gflfof/Desktop/new work/low_rank fcm/preposition_data/";
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-dev", argc, argv)) > 0) strcpy(dev_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(res_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-baseemb", argc, argv)) > 0) strcpy(baseemb_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-freqfile", argc, argv)) > 0) strcpy(freq_file, argv[i + 1]);
    
    if ((i = ArgPos((char *)"-eta", argc, argv)) > 0) eta0 = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-finetuning", argc, argv)) > 0) finetuning = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atof(argv[i + 1]);
    //string dir = "/Users/gflfof/Desktop/new work/low_rank fcm/ppdb_ngram/";
    if (true) {
        PPDBM2M* plearner = new PPDBM2M(baseemb_file, train_file);
        plearner -> adagrad = true;
        plearner -> update_emb = finetuning;
        plearner -> InitVocab(freq_file);
        plearner -> InitSubmodels();
        plearner -> PrintModelInfo();
        
        plearner -> iter = 10;
        plearner -> eta = plearner -> eta0 = 0.01;
        plearner -> lambda = 0;
        plearner -> lambda_prox = 0;
        
        plearner -> EvalLogLoss(dev_file);
        plearner -> EvalMRR(dev_file, THRES1);
        plearner -> EvalMRR(dev_file, THRES2);
        plearner -> TrainData(train_file, dev_file, PREP);
        cout << "Final:" << endl;
        plearner -> EvalLogLoss(dev_file);
        plearner -> EvalMRR(dev_file, THRES1);
        plearner -> EvalMRR(dev_file, THRES2);
        
        cout << "end" << endl;
        return 0;
    }
}
