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

char train_file[MAX_STRING], dev_file[MAX_STRING], output_file[MAX_STRING];
char clus_file[MAX_STRING], baseemb_file[MAX_STRING], freq_file[MAX_STRING];
char model_file[MAX_STRING];
char trainsub_file[MAX_STRING];
char postfix[MAX_STRING];
int iter = 1;
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
    output_file[0] = 0;
    if (argc == 1) {
        ////////////////////////////////////////
        ///       ML2010 Task                ///
        ////////////////////////////////////////
        if (true) {
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.train");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.dev");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.nytsample.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.bllip.word");
            //strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model");
            GetPhraseSim* scorer = new GetPhraseSim("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model.emb");
            
            //scorer -> LoadEmb(baseemb_file, false);
            scorer -> LoadModel(model_file, true);
            scorer -> Init();
            scorer -> fea_model -> InitClusDict(clus_file);
            scorer -> fea_model -> lex_fea = true;
            scorer -> iter = 0;
            
            //scorer -> GetSim("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/adjectivenouns.par", "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/adjectivenouns.res");
            //scorer -> GetSim("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/compoundnouns.par", "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/compoundnouns.res");
            scorer -> GetSim("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/adjectivenouns.par", "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/adjectivenouns.feacomp.res");
            scorer -> GetSim("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/compoundnouns.par", "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/compoundnouns.feacomp.res");
        }
        
        return 0;
    }
    
    
    else{
        //if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-dev", argc, argv)) > 0) strcpy(dev_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-emb", argc, argv)) > 0) strcpy(baseemb_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-clus", argc, argv)) > 0) strcpy(clus_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-freqfile", argc, argv)) > 0) strcpy(freq_file, argv[i + 1]);
        //if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-finetuning", argc, argv)) > 0) finetuning = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
        if ((i = ArgPos((char *)"-model", argc, argv)) > 0) strcpy(model_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-postfix", argc, argv)) > 0) strcpy(postfix, argv[i + 1]);
            
        //strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model");
        //GetPhraseSim* scorer = new GetPhraseSim("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model.emb");
        GetPhraseSim* scorer = new GetPhraseSim(baseemb_file);
         
        cout << "Loading models" << endl;
        //scorer -> LoadEmb(baseemb_file, false);
        //scorer -> LoadEmb(baseemb_file, true);
        //scorer -> LoadModel(baseemb_file, EMB);
        scorer -> LoadModel(model_file, true);
        scorer -> Init();
        scorer -> fea_model -> InitClusDict(clus_file);
        scorer -> fea_model -> lex_fea = true;
            
            //scorer -> GetSim("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/adjectivenouns.par", "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/adjectivenouns.res");
            //scorer -> GetSim("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/compoundnouns.par", "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/compoundnouns.res");
        string outfile = train_file;
        outfile += postfix;
        cout << train_file << "->" << outfile << endl;
        scorer -> GetSim(train_file, outfile);
        outfile = dev_file;
        outfile += postfix;
        cout << dev_file << "->" << outfile << endl;
        scorer -> GetSim(dev_file, outfile);
    }
    
    
    return 0;
}
