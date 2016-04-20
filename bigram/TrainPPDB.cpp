//
//  word2vec.cpp
//  word2vec
//
//  Created by gflfof gflfof on 14-1-22.
//  Copyright (c) 2014年 hit. All rights reserved.
//

//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sstream>

#include "AllLearner.h"

char train_file[MAX_STRING], dev_file[MAX_STRING], test_file[MAX_STRING];
char train_file_s[MAX_STRING], dev_file_s[MAX_STRING];
char clus_file[MAX_STRING], baseemb_file[MAX_STRING], freq_file[MAX_STRING];
char model_file[MAX_STRING];
char trainsub_file[MAX_STRING];
int iter = 20;
int finetuning = 1;
real alpha = 0.01;

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
    train_file_s[0]= '\0';
    if (argc == 1) {
        cout << "./train -train ../nyt_eng_199407.bnp.trainall -trainsub ../nyt_eng_199407.bnp.trainsub -dev ../nyt_eng_199407.bnp.dev -emb ../../model/vectors.nyt.cbow.out -clus ../clusters.nyt.cbow.out.c200 -freqfile ../nyt.freq.txt -model ../nyt_eng_199407.model.FirstDay -iter 1 -alpha 0.01" << endl;
        return 0;
    }
    
    else{
        //if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-test", argc, argv)) > 0) strcpy(test_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-train-s", argc, argv)) > 0) strcpy(train_file_s, argv[i + 1]);
        if ((i = ArgPos((char *)"-trainsub", argc, argv)) > 0) strcpy(trainsub_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-dev", argc, argv)) > 0) strcpy(dev_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-emb", argc, argv)) > 0) strcpy(baseemb_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-clus", argc, argv)) > 0) strcpy(clus_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-freqfile", argc, argv)) > 0) strcpy(freq_file, argv[i + 1]);
        //if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
        if ((i = ArgPos((char *)"-model", argc, argv)) > 0) strcpy(model_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-finetuning", argc, argv)) > 0) finetuning = atoi(argv[i + 1]);
//        if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    }
    cout << "Start" << endl;
    cout << baseemb_file << endl;
    cout << train_file << endl;
            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, "", train_file, train_file_s, NORMEMB);
            
            plearner -> C = 0.1;
            plearner -> norms = (real*)malloc(16 * sizeof(real));
            plearner -> scores = (real*)malloc(16 * sizeof(real));
            plearner -> e_norms = (real*)malloc(16 * plearner->layer1_size * sizeof(real));
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
    
            plearner -> InitVocab(freq_file);
            plearner -> LoadMaintaskVocab(train_file, PPDB_INST);
            plearner -> mask = true;

    plearner -> iter = iter;
    plearner -> eta = plearner -> eta0 = alpha;
    plearner -> update_emb = (finetuning == 1);
    //plearner -> fea_model -> lex_fea = false;
            
        //plearner -> EvalMRR(dev_file, 1000);
        //plearner -> EvalMRR(test_file, 1000);
        //plearner -> EvalMRR(dev_file, 5000);
        plearner -> EvalMRR(dev_file, 10000);
        plearner -> EvalMRR(test_file, 10000);
        //plearner -> EvalMRR(dev_file, 50000);
        //plearner -> EvalMRR(dev_file, 100000);
        //plearner -> EvalMRR(test_file, 100000);
        return 0;
        int start = clock();
        plearner -> next_random = 0;
        plearner -> TrainBigDataNew(train_file, "", dev_file, "", PPDB_INST);
        int now = clock(); 
        cout <<  ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000) << " seconds" << endl;
        //return 0;
        plearner -> EvalMRR(dev_file, 1000);
        plearner -> EvalMRR(test_file, 1000);
        //plearner -> EvalMRR(dev_file, 5000);
        plearner -> EvalMRR(dev_file, 10000);
        plearner -> EvalMRR(test_file, 10000);
        //plearner -> EvalMRR(dev_file, 50000);
        plearner -> EvalMRR(dev_file, 100000);
        plearner -> EvalMRR(test_file, 100000);
        return 0;
        plearner -> iter = 1;
        plearner -> TrainBigDataNew("", train_file_s, dev_file, test_file, PPDB_INST);
        
        plearner -> EvalMRR(dev_file, 1000);
        plearner -> EvalMRR(test_file, 1000);
        //plearner -> EvalMRR(dev_file, 5000);
        plearner -> EvalMRR(dev_file, 10000);
        plearner -> EvalMRR(test_file, 10000);
        //plearner -> EvalMRR(dev_file, 50000);
        plearner -> EvalMRR(dev_file, 100000);
        plearner -> EvalMRR(test_file, 100000);

       /* 
        plearner -> eta = plearner -> eta0 = 0.01;
        plearner -> TrainBigDataNew(train_file, "", dev_file, PPDB_INST);
        plearner -> EvalMRR(dev_file, 1000);
        plearner -> EvalMRR(dev_file, 5000);
        plearner -> EvalMRR(dev_file, 10000);
        plearner -> EvalMRR(dev_file, 50000);
        plearner -> EvalMRR(dev_file, 100000);*/
            free(plearner -> norms);
            free(plearner -> scores);
            free(plearner -> e_norms);
    
    return 0;
}
