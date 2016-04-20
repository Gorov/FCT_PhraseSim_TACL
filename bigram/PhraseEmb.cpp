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
#include <limits>

#include "AllLearner.h"

char train_file[MAX_STRING], dev_file[MAX_STRING], output_file[MAX_STRING];
char clus_file[MAX_STRING], baseemb_file[MAX_STRING], freq_file[MAX_STRING];
char model_file[MAX_STRING];
char feature_file[MAX_STRING];
char trainsub_file[MAX_STRING];
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
        
        if (false) {
            //strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/jair.train");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/jair.data");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.jair.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            //strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.bllip.brown");
            //strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.bllip.word");
            
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.jair.filter.filtered");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.new.jair.filtered");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.fixout.jair.filtered");
            strcpy(feature_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.new.model.fea");
            strcpy(feature_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.fixout.model.fea");
            //strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.fixout.model");
            //strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.new.model");
            
            ClassificationLearnerNew* plearner = new ClassificationLearnerNew();
            //plearner -> LoadModel(model_file);
            //plearner -> LoadModel(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.model.emb");
            //plearner -> LoadModel(baseemb_file, feature_file, LM);
            plearner -> LoadModel(model_file, feature_file, baseemb_file);
            plearner -> fea_model -> InitClusDict(clus_file);
            plearner -> fea_model -> lex_fea = true;
            plearner -> iter = 0;
            
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            
            //plearner -> EvalData(train_file);
            //plearner -> EvalData14(train_file);
            plearner -> EvalData(dev_file);
            plearner -> EvalData14(dev_file);
            plearner -> EvalDataDS(dev_file);
            plearner -> EvalData14DS(dev_file);
            
            cout << "end" << endl;
        }
        
        //ClassificationLearner* plearner = new ClassificationLearner(baseemb_file, train_file);
        /*
         ClassificationLearner* plearner = new ClassificationLearner(baseemb_file, clus_file, train_file);
         cout << "Number of Features: " << plearner -> num_fea << endl;
         */
        
        //ClassificationLearnerNew* plearner = new ClassificationLearnerNew(baseemb_file, clus_file, train_file);
        //MultiBinaryLearner* plearner = new MultiBinaryLearner(baseemb_file, clus_file, train_file);
        //        MultiBinaryLearner* plearner = new MultiBinaryLearner(baseemb_file, train_file);
        //        cout << "Number of Features: " << plearner -> num_fea << endl;
        ////        
        ////        
        //        plearner -> EvalData(train_file);
        //        plearner -> EvalDataAcc(train_file);
        //        plearner -> EvalData(dev_file);
        //        plearner -> EvalData14(dev_file);
        //        plearner -> EvalDataAcc(dev_file);
        //        plearner -> TrainData(train_file);
        //        plearner -> EvalData(train_file);
        //        plearner -> EvalDataAcc(train_file);
        //        plearner -> EvalData(dev_file);
        //        plearner -> EvalData14(dev_file);
        //        plearner -> EvalDataAcc(dev_file);
        
        ////////////////////////////////////////
        ///       Binary   Training          ///
        ////////////////////////////////////////
        
        if (false) {
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.trainSet");
            //strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.trial");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.testSet");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.semeval2013.filtered");
            //strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.nytsample.filtered");
            //strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model.emb");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
//            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.bllip.word");
//            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.bllip.brown");
            
            //        //BinaryLearner* plearner = new BinaryLearner(baseemb_file, train_file);
            //        //BinaryLearner* plearner = new BinaryLearner(baseemb_file, clus_file, train_file);
            //        //BinaryLearnerNew* plearner = new BinaryLearnerNew(baseemb_file, clus_file, train_file);
            BinaryThresLearner* plearner = new BinaryThresLearner(baseemb_file, clus_file, train_file);
            //        //BinaryThresLearner* plearner = new BinaryThresLearner(baseemb_file, train_file);
            plearner -> fea_model -> SaveModel("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/tmp.model.fea");
            cout << "Number of Features: " << plearner -> num_fea << endl;
            plearner -> iter = 20;
            plearner -> eta = plearner -> eta0 = 0.01;
            plearner -> update_emb = true;
            
            plearner -> BuildVocab(train_file);
            
            plearner -> EvalData(train_file);
            plearner -> EvalData(dev_file);
            plearner -> TrainData(train_file);
            plearner -> EvalData(train_file);
            plearner -> EvalData(dev_file);
            //plearner -> EvalDataObs(train_file);
            //plearner -> EvalDataObs(dev_file);
            return 0;
        }
        
        ////////////////////////////////////////
        ///       MTL      Training          ///
        ////////////////////////////////////////
        char train_file_b[MAX_STRING], dev_file_b[MAX_STRING];
        char train_file_p[MAX_STRING], dev_file_p[MAX_STRING];
        char train_file_s[MAX_STRING], dev_file_s[MAX_STRING];
        if (false) {
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.trainSet");
            //strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.trial");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.testSet");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.semeval2013.filtered");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.ppdb.semeval2013.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            
            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, train_file, "", train_file, HALFLM);
            cout << "Number of Features: " << plearner -> num_fea << endl;
            plearner -> iter = 20;
            plearner -> eta = plearner -> eta0 = 0.01;
            plearner -> update_emb = true;
            
            plearner -> EvalData(train_file, BINARY_INST);
            plearner -> EvalData(dev_file, BINARY_INST);
            plearner -> TrainData(train_file, "", train_file);
            plearner -> EvalData(train_file, BINARY_INST);
            plearner -> EvalData(dev_file, BINARY_INST);
            return 0;
        }
        
        if (false) {
            strcpy(train_file_b, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.trainSet");
            strcpy(dev_file_b, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.testSet");
            strcpy(train_file_p, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.train");
            strcpy(dev_file_p, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.dev");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.ppdb.filtered");
            //strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.ppdbb2u.filtered");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.ppdb.semeval2013.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            
            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, train_file_b, train_file_p, "", HALFLM);
            
            plearner -> eta = plearner -> eta0 = 0.01;
            plearner -> iter = 20;
            plearner -> update_emb = true;
            //plearner -> update_emb = false;
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            
            plearner -> EvalData(train_file_b, BINARY_INST);
            plearner -> EvalData(dev_file_b, BINARY_INST);
            plearner -> EvalMRR(dev_file_p, 1000);
            plearner -> next_random = 0;
            plearner -> TrainData(train_file_b, train_file_p, "");
            
            plearner -> EvalData(train_file_b, BINARY_INST);
            plearner -> EvalData(dev_file_b, BINARY_INST);
            //plearner -> EvalMRR(train_file_p, 1000);
            plearner -> EvalMRR(dev_file_p, 1000);
            return 0;
        }
        
        if (false) {
            strcpy(train_file_b, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.trainSet");
            strcpy(train_file_b, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.trainSet.copy5");
            strcpy(dev_file_b, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.testSet");
            strcpy(train_file_s, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.train");
            strcpy(dev_file_s, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.dev");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.nyt.semeval2013.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            
            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, train_file_b, "", train_file_s, LM);
            //            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, train_file_b, "", "", LM);
            plearner -> syn0 = (real*)malloc(sizeof(real) * plearner -> layer1_size * plearner -> emb_model -> vocab_size);
            memcpy(plearner -> syn0, plearner -> emb_model -> syn0, sizeof(real) * plearner -> layer1_size * plearner -> emb_model -> vocab_size);
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            
            plearner -> EvalData(dev_file_b, BINARY_INST_DS);
            plearner -> LoadMaintaskVocab(train_file_b, BINARY_INST_DS);
            plearner -> mask = true;
            
            plearner -> eta = plearner -> eta0 = 0.01;
            plearner -> next_random = 0;
            plearner -> iter = 3;
            plearner -> C = 0.1;
            plearner -> update_emb = true;
            
            plearner -> TrainData(train_file_b, "", "");
            //            plearner -> EvalData(dev_file_b, BINARY_INST);
            //plearner -> TrainDataNew(train_file_b, train_file_s);
            //            plearner -> TrainBigData(train_file_b, train_file_s, dev_file_b);
            //            plearner -> TrainDataNew(train_file_b, "");
            //            plearner -> EvalData(dev_file_b, BINARY_INST);
            //            plearner -> TrainDataNew(train_file_b, train_file_s);
            //            plearner -> EvalData(dev_file_b, BINARY_INST);
            //            plearner -> TrainData(train_file_b, "", "");
            //            plearner -> TrainDataNew(train_file_b, "");
            //            plearner -> TrainData(train_file_b, "", "");
            //plearner -> EvalLogLoss(train_file_s);
            //plearner -> EvalLogLoss(dev_file_s);
            plearner -> EvalData(train_file_b, BINARY_INST_DS);
            plearner -> EvalData(dev_file_b, BINARY_INST_DS);
            
            return 0;
        }
        
        if (true) {
            strcpy(train_file_p, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.train");
//            strcpy(train_file_p, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.train.trial");
            //strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.test");
            strcpy(dev_file_p, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.dev");
            strcpy(train_file_s, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.train");
            strcpy(dev_file_s, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.dev");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.ppdb.filtered");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.ppdbb2u.filtered");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.nyt.ppdb.filtered");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.new.model.nyt.ppdb.emb");
//            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.new.model.emb");
            
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.model");
            
            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, "", train_file_p, train_file_s, LM);
//            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, "", train_file_p, "", LM);
            
            plearner -> eta = plearner -> eta0 = 0.05;
            plearner -> iter = 5;
            plearner -> update_emb = true;
            plearner -> C = 0.1;
            //plearner -> update_emb = false;
            plearner -> norms = (real*)malloc(16 * sizeof(real));
            plearner -> scores = (real*)malloc(16 * sizeof(real));
            plearner -> e_norms = (real*)malloc(16 * plearner->layer1_size * sizeof(real));
//            for (int a = 0; a < plearner -> layer1_size; a++) plearner -> fea_model -> b1s[a] = 0.5;
//            for (int a = 0; a < plearner -> layer1_size; a++) plearner -> fea_model -> b2s[a] = 0.5;
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            
//            plearner -> EvalMRR(train_file_p, 1000);
            plearner -> EvalMRR(dev_file_p, 1000);
//            plearner -> EvalMRR(train_file_p, 5000);
//            plearner -> EvalMRR(dev_file_p, 5000);
            //return 0;
            plearner -> next_random = 0;
//            plearner -> TrainBigDataNew(train_file_p, train_file_s, dev_file_p, PPDB_INST);
            plearner -> TrainBigDataNew(train_file_p, "", dev_file_p, PPDB_INST);
//            plearner -> TrainData("", train_file_p, "");
            //plearner -> TrainData(train_file, train_file, dev_file);
//            plearner -> EvalMRR(train_file_p, 1000);
            plearner -> EvalMRR(dev_file_p, 1000);
            //plearner -> EvalMRR(train_file_p, 5000);
//            plearner -> EvalMRR(train_file_p, 5000);
            plearner -> EvalMRR(dev_file_p, 5000);
            free(plearner -> norms);
            free(plearner -> scores);
            free(plearner -> e_norms);
            return 0;
        }
        
        if (false) {
            strcpy(train_file_b, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.trainSet");
            strcpy(train_file_b, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.trainSet.copy5");
            strcpy(dev_file_b, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/en.testSet");
            strcpy(train_file_s, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.train");
            strcpy(dev_file_s, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.dev");
            
//            strcpy(train_file_s, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.train");
//            strcpy(dev_file_s, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.dev");
//            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.nyt.filtered");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.nyt.semeval2013.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            
            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, train_file_b, "", train_file_s, LM);
//            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, train_file_b, "", "", LM);
            plearner -> syn0 = (real*)malloc(sizeof(real) * plearner -> layer1_size * plearner -> emb_model -> vocab_size);
            memcpy(plearner -> syn0, plearner -> emb_model -> syn0, sizeof(real) * plearner -> layer1_size * plearner -> emb_model -> vocab_size);
//            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, train_file_b, "", "", LM);
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");

//            plearner -> EvalLogLoss(train_file_s);
//            plearner -> EvalLogLoss(dev_file_s);
            plearner -> EvalData(dev_file_b, BINARY_INST_DS);
            plearner -> LoadMaintaskVocab(train_file_b, BINARY_INST_DS);
            plearner -> mask = true;
            
            plearner -> eta = plearner -> eta0 = 0.01;
            plearner -> next_random = 0;
            plearner -> iter = 1;
            plearner -> C = 0.01;
            plearner -> update_emb = true;
            
            plearner -> TrainData(train_file_b, "", "");
            plearner -> EvalData(dev_file_b, BINARY_INST_DS);
            //plearner -> TrainDataNew(train_file_b, train_file_s);
            plearner -> TrainBigData(train_file_b, train_file_s, dev_file_b);
//            plearner -> TrainBigData(train_file_b, "", dev_file_b);
//            plearner -> TrainDataNew(train_file_b, "");
            plearner -> EvalData(dev_file_b, BINARY_INST_DS);
//            plearner -> TrainDataNew(train_file_b, train_file_s);
//            plearner -> EvalData(dev_file_b, BINARY_INST);
//            plearner -> TrainData(train_file_b, "", "");
//            plearner -> TrainDataNew(train_file_b, "");
            plearner -> TrainData(train_file_b, "", "");
            //plearner -> EvalLogLoss(train_file_s);
            //plearner -> EvalLogLoss(dev_file_s);
            plearner -> EvalData(train_file_b, BINARY_INST_DS);
            plearner -> EvalData(dev_file_b, BINARY_INST_DS);
            
            return 0;
        }
        
        ////////////////////////////////////////
        ///       Skipgram Training          ///
        ////////////////////////////////////////
        if (false) {
            strcpy(train_file_s, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.train");
            strcpy(dev_file_s, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.dev");
            
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.nyt.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            
            MTLTrainer* plearner = new MTLTrainer(baseemb_file, clus_file, "", "", train_file_s, LM);
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            
            for (int a = 0; a < plearner -> layer1_size; a++) plearner -> fea_model -> b1s[a] = 0.5;
            for (int a = 0; a < plearner -> layer1_size; a++) plearner -> fea_model -> b2s[a] = 0.5;
            plearner -> e2_norm = (real*)malloc(plearner->layer1_size * sizeof(real));
            
            plearner -> EvalData(train_file_s, SKIPGRAM_INST_DS);
            plearner -> EvalData(dev_file_s, SKIPGRAM_INST_DS);
            plearner -> EvalLogLoss(dev_file_s);
            
            plearner -> eta = plearner -> eta0 = 0.01;
            plearner -> next_random = 0;
            plearner -> iter = 1;
            plearner -> C = 1;
            plearner -> update_emb = true;
            
            plearner -> TrainDataNew(train_file_s);
            plearner -> EvalData(train_file_s, SKIPGRAM_INST_DS);
            plearner -> EvalData(dev_file_s, SKIPGRAM_INST_DS);
            plearner -> EvalLogLoss(dev_file_s);
            
            return 0;
        }
        
        if (false) {
        strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.train");
        strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.dev");
        
//        strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.train");
//        strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.dev");
        strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.nytsample.filtered");
        strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.nyt.filtered");
        //strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.bllip.word");
        strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
        strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model");
        
        SkipgramLearner* plearner = new SkipgramLearner(baseemb_file, clus_file, train_file, LM);
        //SkipgramLearner* plearner = new SkipgramLearner(baseemb_file, train_file);
        plearner -> negative = 15;
//        SkipgramLearner* plearner = new SkipgramLearner(baseemb_file);
//        plearner -> unnorm = false;
//        plearner -> halflm = true;
//        plearner -> InitWithOption(baseemb_file, clus_file, train_file);
//        
//        plearner -> iter = 1;
//        plearner -> update_emb = true;
//        
//        //        SkipgramLearner* plearner = new SkipgramLearner("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model.emb");
//        //        plearner -> LoadModel(model_file);
//        //        plearner -> fea_model -> InitClusDict(clus_file);
//        //        plearner -> fea_model -> lex_fea = true;
//        //        plearner -> iter = 0;
//        
        cout << "Number of Features: " << plearner -> num_fea << endl;
        cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
        plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
//        //plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.test.freq.txt");
//        
//        //cout << plearner -> fea_model -> b1s[0] << endl;
//        //cout << plearner -> fea_model -> b2s[0] << endl;
//        plearner -> EvalData(train_file);
//        plearner -> EvalData(dev_file);
//            for (int a = 0; a < plearner -> layer1_size; a++) plearner -> fea_model -> b1s[a] = 0.5;
//            for (int a = 0; a < plearner -> layer1_size; a++) plearner -> fea_model -> b2s[a] = 0.5;
            plearner -> EvalLogLoss(train_file);
        plearner -> EvalLogLoss(dev_file);
//        plearner -> EvalLogLossBase(train_file);
//        plearner -> EvalLogLossBase(dev_file);
        plearner -> next_random = 0;
            plearner -> iter = 20;
//        plearner -> TrainData(train_file);
            plearner -> TrainBigData(train_file, train_file, dev_file);
        //plearner -> TrainData(train_file, train_file, dev_file);
        plearner -> EvalData(train_file);
        plearner -> EvalData(dev_file);
        plearner -> EvalLogLoss(train_file);
        plearner -> EvalLogLoss(dev_file);
//        plearner -> EvalLogLossBase(train_file);
//        plearner -> EvalLogLossBase(dev_file);
         //plearner -> fea_model -> SaveModel("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/tmp.model.fea2");
//        plearner -> SaveModel("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model", false);
            return 0;
        }
        //hs training
        if (false) {
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.train");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.dev");
            
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.train");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.dev");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.nytsample.filtered");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.nyt.filtered");
            //strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.bllip.word");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model");
            
            SkipgramLearner* plearner = new SkipgramLearner(baseemb_file, clus_file, train_file, LM);
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            plearner -> BuildBinaryTree();
            plearner -> InitExpTable();
            plearner -> emb_model -> InitNetHS();
            plearner -> negative = 0;
            plearner -> hs = 1;
            
            plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
            plearner -> EvalLogLossBase(train_file);
            plearner -> EvalLogLossBase(dev_file);
            plearner -> next_random = 0;
            plearner -> iter = 20;
            plearner -> TrainData(train_file);
            //plearner -> TrainData(train_file, train_file, dev_file);
            plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
            plearner -> EvalLogLossBase(train_file);
            plearner -> EvalLogLossBase(dev_file);
            return 0;
        }
        //hs load model
        if (false) {
            strcpy(freq_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.train");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.dev");
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.train");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.dev");
            SkipgramLearner* plearner = new SkipgramLearner();
            plearner -> emb_model = new EmbeddingModel("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.model4.emb", HSLM);
            //plearner -> emb_model = new EmbeddingModel("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.model2", HSLM);
            
            plearner -> fea_model = new FeatureModel();
            plearner -> fea_model -> LoadModel("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.model4.fea");
            plearner -> fea_model -> InitClusDict(clus_file);
            plearner -> fea_model -> lex_fea = true;
            
//            plearner -> fea_model = new FeatureModel(plearner -> emb_model -> layer1_size, train_file, clus_file, 2);
            
            plearner -> layer1_size = plearner -> emb_model -> layer1_size;
            plearner -> num_fea = plearner -> fea_model -> num_fea;
            ((SkipgramLearner*)plearner) -> Init();
            ((BaseLearner*)(plearner)) -> Init();
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.freq1");
            plearner -> BuildBinaryTree();
            plearner -> InitExpTable();
            plearner -> negative = 0;
            plearner -> hs = 1;
            plearner -> eta0 = 0.01;
            
            //plearner -> InitTrainInfoLM();
            
            plearner -> update_emb = true;
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            
            cout << plearner -> fea_model -> b1s[0] << endl;
            cout << plearner -> fea_model -> b2s[0] << endl;
            //plearner -> SaveModel("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.model2.new", true, HSLM);
            //plearner -> EvalData(train_file);
            //plearner -> EvalData(dev_file);
            plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
            plearner -> next_random = 0;
            plearner -> iter = 1;
            plearner -> TrainData(train_file);
            //plearner -> EvalData(train_file);
            //plearner -> EvalData(dev_file);
            plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
            
            //plearner -> SaveModel("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.model4", true, HSLM);
            return 0;
        }
        
        ////////////////////////////////////////
        ///       PPDB  B2U  Training        ///
        ////////////////////////////////////////
        if (false) {
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.train");
            //strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.test");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.dev");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.ppdb.filtered");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.ppdbb2u.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.model");
            
            PPDBLearnerB2U* plearner = new PPDBLearnerB2U(baseemb_file, clus_file, train_file, LM);
            
            plearner -> eta = plearner -> eta0 = 0.01;
            plearner -> iter = 5;
            plearner -> update_emb = true;
            //plearner -> update_emb = false;
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            
            plearner -> BuildVocab(train_file);
            
            plearner -> EvalData(train_file);
            plearner -> EvalData(dev_file);
            //plearner -> EvalLogLoss(train_file);
            //plearner -> EvalLogLoss(dev_file);
            plearner -> EvalMRR(dev_file, 1000);
            plearner -> next_random = 0;
            plearner -> TrainData(train_file);
            //plearner -> TrainData(train_file, train_file, dev_file);
            plearner -> EvalData(train_file);
            plearner -> EvalData(dev_file);
            //plearner -> EvalDataObs(train_file);
            //plearner -> EvalDataObs(dev_file);
            //plearner -> EvalLogLoss(train_file);
            //plearner -> EvalLogLoss(dev_file);
            plearner -> EvalMRR(dev_file, 1000);
            plearner -> EvalMRRObs(dev_file, 1000);
            //plearner -> EvalMRR(train_file, 1000);
            plearner -> SaveModel(model_file);
        }
        if (false) {
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.train");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.dev");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.ppdb.filtered");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.ppdbb2u.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.model");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.fixout.model");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.new.model");
            
            PPDBLearnerB2U* plearner = new PPDBLearnerB2U("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.model.emb");
            //plearner -> LoadModel(model_file);
            //plearner -> LoadModel(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.model.emb");
            plearner -> LoadModel(model_file, baseemb_file);
            //plearner -> LoadEmb(baseemb_file);
            plearner -> fea_model -> InitClusDict(clus_file);
            plearner -> fea_model -> lex_fea = true;
            plearner -> iter = 5;
            
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            //plearner -> EvalData(dev_file);
            plearner -> EvalMRR(dev_file, 1000);
//            plearner -> EvalMRRDouble(dev_file, 1000);
            plearner -> EvalMRR(dev_file, 5000);
//            plearner -> EvalMRRDouble(dev_file, 5000);
            
            plearner -> eta = plearner -> eta0 = 0.005;
            plearner -> next_random = 0;
            plearner -> TrainData(train_file);
            //plearner -> EvalData(train_file);
            //plearner -> EvalData(dev_file);
            plearner -> EvalMRR(dev_file, 1000);
            plearner -> EvalMRR(dev_file, 5000);
//            for (int i = 2000; i < 5000; i+=1000) {
//                plearner -> EvalMRR(dev_file, i);
//                plearner -> EvalMRRDouble(dev_file, i);
//            }
            return 0;
        }
        if (false) {
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.train");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.xxl.b2u.dev");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.ppdbb2u.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.new.model");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt1.fixout.model");
            
            //PPDBLearnerB2U* plearner = new PPDBLearnerB2U("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.model.emb");
            PPDBLearnerB2UDoubleSpace* plearner = new PPDBLearnerB2UDoubleSpace("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.model.emb");
            plearner -> LoadModel(model_file, baseemb_file);
            plearner -> fea_model -> InitClusDict(clus_file);
            plearner -> fea_model -> lex_fea = true;
            plearner -> eta = plearner -> eta0 = 0.01;
            plearner -> iter = 5;
            plearner -> update_emb = true;
            //plearner -> update_emb = false;
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            
            //plearner -> EvalData(train_file);
            //plearner -> EvalData(dev_file);
            plearner -> EvalMRR(dev_file, 1000);
            plearner -> EvalMRRDouble(dev_file, 1000);
            plearner -> next_random = 0;
            plearner -> TrainData(train_file);
            //plearner -> EvalData(train_file);
            //plearner -> EvalData(dev_file);
            plearner -> EvalMRR(dev_file, 1000);
            plearner -> EvalMRRDouble(dev_file, 1000);
        }
        
        ////////////////////////////////////////
        ///       PPDB     Training          ///
        ////////////////////////////////////////
        if (false) {
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.test.data");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.dev.data");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.ppdbtest.filtered");
            //strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.bllip.word");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model");
            
            PPDBLearner* plearner = new PPDBLearner(baseemb_file, clus_file, train_file);
            
            plearner -> eta = plearner -> eta0 = 0.01;
            plearner -> iter = 10;
            plearner -> update_emb = false;
            //plearner -> update_emb = false;
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            
            plearner -> EvalData(train_file);
            plearner -> EvalData(dev_file);
            plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
            plearner -> next_random = 0;
            plearner -> TrainData(train_file);
            //plearner -> TrainData(train_file, train_file, dev_file);
            plearner -> EvalData(train_file);
            plearner -> EvalData(dev_file);
            plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
            plearner -> SaveModel(model_file);
        }
        if (false) {
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.test.data");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.dev.data");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.ppdbtest.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model");
            
            PPDBLearner* plearner = new PPDBLearner("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model.emb");
            plearner -> LoadModel(model_file);
            plearner -> fea_model -> InitClusDict(clus_file);
            plearner -> fea_model -> lex_fea = true;
            plearner -> iter = 0;
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            
            //plearner -> EvalData(train_file);
            //plearner -> EvalData(dev_file);
            //plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
        }
        if (false) {
            strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.test.data");
            strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/ppdb.dev.data");
            strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.cbow.out.ppdbtest.filtered");
            strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
            strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model");
            
            MRRScorer* scorer = new MRRScorer("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model.emb");
            scorer -> LoadModel(model_file);
            //scorer -> LoadModel(baseemb_file, true);
            //scorer -> LoadModel("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model.emb", true);
            scorer -> fea_model -> InitClusDict(clus_file);
            scorer -> fea_model -> lex_fea = true;
            scorer -> iter = 0;
            
            cout << "Number of Features: " << scorer -> num_fea << endl;
            cout << "Number of Instances: " << scorer -> fea_model -> num_inst << endl;
            
            scorer -> EvalMRR(dev_file, 1000, dev_file, false);
            //scorer -> EvalLogLoss(dev_file);
        }
        
        
        ////////////////////////////////////////
        ///       ML2010 Task                ///
        ////////////////////////////////////////
        if (false) {
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
        
        ////////////////////////////////////////
        ///       Joint    Training          ///
        ////////////////////////////////////////
        
        char train_lm_file[MAX_STRING];
        char freq_file[MAX_STRING];
        //strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.train");
        //strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.sample.dev");
        strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.train");
        strcpy(dev_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.dev");
        strcpy(train_lm_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.tok");
        strcpy(train_lm_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.tok");
        //strcpy(train_lm_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.trial.tok");
        strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/vectors.nyt.skipgram.ncelm.nyt.filtered");
        strcpy(clus_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/clusters.nyt.cbow.out.c200");
        
        if (false) {
            //SkipgramLearner* plearner = new SkipgramLearner(baseemb_file, clus_file, train_file);
            
            //strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.model");
            //strcpy(baseemb_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model");
            JointLearner* plearner = new JointLearner(baseemb_file, clus_file, train_file, LM);
            ((SkipgramLearner*)plearner) -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
//            plearner -> vocab_size = plearner -> emb_model -> vocab_size;
//            for (int b = 0; b < plearner->vocab_size; b++) {
//                for (int a = 0; a < plearner->layer1_size; a++) plearner -> emb_model-> syn1neg[a + b * plearner->layer1_size] = 0.0;
//            }

            plearner -> InitTrainer(0.001, 1e-3, 5, 15, train_lm_file, 8);
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            //        
            cout << plearner -> fea_model -> b1s[0] << endl;
            cout << plearner -> fea_model -> b2s[0] << endl;
            plearner -> EvalData(train_file);
            plearner -> EvalData(dev_file);
            plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
            plearner -> next_random = 0;
            plearner -> iter = 20;
            plearner -> JointTrainBigData(train_file, train_lm_file, trainsub_file, dev_file);
            //plearner -> TrainData(train_file);
            plearner -> EvalData(train_file);
            plearner -> EvalData(dev_file);
            plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
            //        plearner -> EvalLogLossBase(train_file);
            //        plearner -> EvalLogLossBase(dev_file);
            return 0;
        }
        
        if (false) {
            strcpy(freq_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
            JointLearner* plearner = new JointLearner();
            plearner -> emb_model = new EmbeddingModel();
            plearner -> emb_model -> layer1_size = 200;
            plearner -> emb_model -> InitFromTrainFile(train_lm_file);
            plearner -> fea_model = new FeatureModel(plearner -> emb_model -> layer1_size, train_file, clus_file, 2);
            plearner -> layer1_size = plearner -> emb_model -> layer1_size;
            plearner -> num_fea = plearner -> fea_model -> num_fea;
            ((SkipgramLearner*)plearner) -> Init();
            ((BaseLearner*)(plearner)) -> Init();
            
            plearner -> InitVocab();
            plearner -> BuildBinaryTree();
            plearner -> emb_model -> InitNetHS();
            
            plearner -> InitTrainer(0.025, 1e-3, 5, 15, train_lm_file, 1);
            plearner -> negative = 0;
            plearner -> hs = 1;
            
            //plearner -> InitTrainInfoLM();
            
            plearner -> update_emb = true;
            
            cout << "Number of Features: " << plearner -> num_fea << endl;
            cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
            
            //plearner -> EvalData(train_file);
            //plearner -> EvalData(dev_file);
            plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
            plearner -> next_random = 0;
            plearner -> iter = 10;
            plearner -> JointTrainBigData(train_file, train_lm_file, trainsub_file, dev_file);
            //plearner -> TrainData(train_file);
            //plearner -> EvalData(train_file);
            //plearner -> EvalData(dev_file);
            plearner -> EvalLogLoss(train_file);
            plearner -> EvalLogLoss(dev_file);
            //plearner -> emb_model -> SaveLM("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model");
            
            FILE* fo = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.model2", "w");
                // Save the word vectors
                fprintf(fo, "%lld %lld\n", plearner -> vocab_size, plearner -> layer1_size);
                for (int a = 0; a < plearner -> vocab_size; a++) {
                    fprintf(fo, "%s ", plearner -> vocab[a].c_str());
                    for (int b = 0; b < plearner -> layer1_size; b++) fprintf(fo, "%lf ", plearner -> emb_model-> syn0[a * plearner -> layer1_size + b]);
                    fprintf(fo, "\n");
                }
            for (int a = 0; a < plearner -> vocab_size - 1; a++) {
                    for (int b = 0; b < plearner -> layer1_size; b++) fprintf(fo, "%lf ", plearner -> emb_model->syn1[a * plearner -> layer1_size + b]);
                    fprintf(fo, "\n");
                }
            fclose(fo);
            return 0;
        }
        
        if (false){
        //strcpy(model_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model");
        
        strcpy(freq_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
                
        //SkipgramLearner* plearner = new SkipgramLearner(baseemb_file, clus_file, train_file);
            JointLearner* plearner = new JointLearner(freq_file, clus_file, train_file, true);
        //SkipgramLearner* plearner = new SkipgramLearner(freq_file, clus_file, train_file, true);
            //plearner -> emb_model -> LoadLMAdditively(baseemb_file);
            plearner -> InitTrainer(0.001, 1e-3, 5, 15, train_lm_file, 10);
            plearner -> InitTrainInfoLM();
        
        plearner -> update_emb = true;

        cout << "Number of Features: " << plearner -> num_fea << endl;
        cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
        ((SkipgramLearner*)plearner) -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
        
        //        
        //        //cout << plearner -> fea_model -> b1s[0] << endl;
        //        //cout << plearner -> fea_model -> b2s[0] << endl;
        plearner -> EvalData(train_file);
        plearner -> EvalData(dev_file);
        plearner -> EvalLogLoss(train_file);
        plearner -> EvalLogLoss(dev_file);
        plearner -> next_random = 0;
        plearner -> iter = 10;
        //plearner -> TrainData(train_file);
            plearner -> JointTrainBigData(train_file, train_lm_file, trainsub_file, dev_file);
        plearner -> EvalData(train_file);
        plearner -> EvalData(dev_file);
        plearner -> EvalLogLoss(train_file);
        plearner -> EvalLogLoss(dev_file);
        //        plearner -> EvalLogLossBase(train_file);
        //        plearner -> EvalLogLossBase(dev_file);
        
        //        plearner -> SaveModel("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model", false);
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
        //        if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
        //        if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
        //        if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
        //        if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
        //        if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
        //        if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
        //        if ((i = ArgPos((char *)"-ppfile", argc, argv)) > 0) strcpy(pp_file, argv[i + 1]);
        //        if ((i = ArgPos((char *)"-lambda", argc, argv)) > 0) lambda = atof(argv[i + 1]);
    }
    
    SkipgramLearner* plearner = new SkipgramLearner(baseemb_file, clus_file, train_file);
    //SkipgramLearner* plearner = new SkipgramLearner(baseemb_file, train_file);
    
    //        SkipgramLearner* plearner = new SkipgramLearner("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.model.emb");
    //        plearner -> LoadModel(model_file);
    //        plearner -> fea_model -> InitClusDict(clus_file);
    //        plearner -> fea_model -> lex_fea = true;
    //        plearner -> iter = 0;
    
    cout << "Number of Features: " << plearner -> num_fea << endl;
    //plearner -> InitVocab("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt.freq.txt");
    plearner -> InitVocab(freq_file);
    plearner -> update_emb = (finetuning == 1);
    
    plearner -> EvalData(train_file);
    plearner -> EvalData(dev_file);
    plearner -> TrainData(train_file);
    plearner -> EvalData(train_file);
    plearner -> EvalData(dev_file);
    
    plearner -> SaveModel(model_file, false);
    
    return 0;
}