//
//  BaseLearner.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-4-25.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "BaseLearner.h"

void BaseLearner::Init() {
    eta = eta0 = 0.01;
    iter = 1;
    /*
    lambda1 = new double[layer1_size];
    lambda2 = new double[layer1_size];
    emb_p = new double[layer1_size];
    
    part_lambda1 = new double[layer1_size];
    part_lambda2 = new double[layer1_size];
    part_emb_p = new double[layer1_size];
    */
    lambda1 = new real[layer1_size];
    lambda2 = new real[layer1_size];
    emb_p = new real[layer1_size];
    
    part_lambda1 = new real[layer1_size];
    part_lambda2 = new real[layer1_size];
    part_emb_p = new real[layer1_size];
}

void BaseLearner::Init(char* embfile, char* trainfile)
{
    //eta = 0.001;
    eta = eta0 = 0.01;
    iter = 20;
    
    emb_model = new EmbeddingModel(embfile);
    //dim = (int)emb_model -> layer1_size;
    layer1_size = emb_model -> layer1_size;
    
    lambda1 = new real[layer1_size];
    lambda2 = new real[layer1_size];
    emb_p = new real[layer1_size];
    
    part_lambda1 = new real[layer1_size];
    part_lambda2 = new real[layer1_size];
    part_emb_p = new real[layer1_size];
    
    update_emb = false;
    five_choice = false;
}

void BaseLearner::Init(char* embfile, char* clusfile, char* trainfile)
{
    eta = eta0 = 0.01;
    iter = 1;
    
    emb_model = new EmbeddingModel(embfile);
    layer1_size = emb_model -> layer1_size;
    
    lambda1 = new real[layer1_size];
    lambda2 = new real[layer1_size];
    emb_p = new real[layer1_size];
    
    part_lambda1 = new real[layer1_size];
    part_lambda2 = new real[layer1_size];
    part_emb_p = new real[layer1_size];
    
    update_emb = true;
    five_choice = false;
}

void BaseLearner::Init(char* embfile, char* clusfile, char* trainfile, int type)
{
    eta = eta0 = 0.01;
    iter = 1;
    
    emb_model = new EmbeddingModel(embfile, type);
    layer1_size = emb_model -> layer1_size;
    
    lambda1 = new real[layer1_size];
    lambda2 = new real[layer1_size];
    emb_p = new real[layer1_size];
    
    part_lambda1 = new real[layer1_size];
    part_lambda2 = new real[layer1_size];
    part_emb_p = new real[layer1_size];
    
    update_emb = true;
    five_choice = false;
}

void BaseLearner::Init(char* freq_file, char* clusfile, char* trainfile, bool random_emb)
{
    eta = eta0 = 0.01;
    iter = 1;
    
    emb_model = new EmbeddingModel(freq_file, RANDEMB);
    layer1_size = emb_model -> layer1_size;
    
    lambda1 = new real[layer1_size];
    lambda2 = new real[layer1_size];
    emb_p = new real[layer1_size];
    
    part_lambda1 = new real[layer1_size];
    part_lambda2 = new real[layer1_size];
    part_emb_p = new real[layer1_size];
    
    update_emb = true;
    five_choice = false;
}

void BaseLearner::ForwardProp()
{
    int a, c;
    long long l1;
    real sum;
    
    ExtractFeature();
    ComputeLambdas();
    
    for (a = 0; a < layer1_size; a++) emb_p[a] = 0.0;
    word2int::iterator iter = emb_model -> vocabdict.find(inst->word1);
    if (iter != emb_model -> vocabdict.end()) {
        inst -> id1 = iter -> second;
        l1 = iter->second * layer1_size;
        for (a = 0; a < layer1_size; a++) emb_p[a] = lambda1[a] * emb_model->syn0[a + l1];
    }
    else inst -> id1 = -1;
    iter = emb_model -> vocabdict.find(inst->word2);
    if (iter != emb_model -> vocabdict.end()) {
        inst -> id2 = iter -> second;
        l1 = iter->second * layer1_size;
        for (a = 0; a < layer1_size; a++) emb_p[a] += lambda2[a] * emb_model->syn0[a + l1];
    }
    else inst -> id2 = -1;
    ForwardOutputs();
}

void BaseLearner::BackProp()
{
    int a;
    long long l1;
    for (a = 0; a < layer1_size; a++) part_emb_p[a] = 0.0;
    
    //Backprop output layer emb
    //if (updata_emb) {
    //    BackPropEmb();
    //}
    long tmpid = BackPropPhrase();
    if (tmpid < 0) return;
    
    for (a = 0; a < layer1_size; a++) part_lambda1[a] = 0.0;
    for (a = 0; a < layer1_size; a++) part_lambda2[a] = 0.0;
    if (inst -> id1 >= 0) {
        l1 = inst -> id1 * layer1_size;
        for (a = 0; a < layer1_size; a++) {
            part_lambda1[a] = part_emb_p[a] * emb_model->syn0[a + l1];
        }
        if (update_emb) for (a = 0; a < layer1_size; a++) {
            emb_model->syn0[a + l1] += eta * part_emb_p[a] * lambda1[a];
        }
    }
    if (inst -> id2 >= 0) {
        l1 = inst -> id2 * layer1_size;
        for (a = 0; a < layer1_size; a++) {
            part_lambda2[a] = part_emb_p[a] * emb_model->syn0[a + l1];
        }
        if (update_emb) for (a = 0; a < layer1_size; a++) {
            emb_model->syn0[a + l1] += eta * part_emb_p[a] * lambda2[a];
        }
    }
    //Backprop feature weights
    BackPropFeatures();
}

void BaseLearner::ExtractFeature() {
    int a[2];
    fea_model -> ExtractFeaBNP(inst, feat_vec1, feat_vec2, a);
    num_feat1 = a[0];
    num_feat2 = a[1];
}

void BaseLearner::ComputeLambdas() {
    int a, c;
    for (c = 0; c < layer1_size; c++) {
        lambda1[c] = 0.0;
        for (a = 0; a < num_feat1; a++) {
            lambda1[c] += fea_model -> param[c * num_fea + feat_vec1[a]];
        }
        //lambda1[c] += fea_model -> b1;
        lambda1[c] += fea_model -> b1s[c];
        //lambda1[c] = 1.0;
    }
    for (c = 0; c < layer1_size; c++) {
        lambda2[c] = 0.0;
        for (a = 0; a < num_feat2; a++) {
            lambda2[c] += fea_model -> param[c * num_fea + feat_vec2[a]];
        }
        //lambda2[c] += fea_model -> b2;
        lambda2[c] += fea_model -> b2s[c];
        //lambda2[c] = 1.0;
    }
    //may need to add bias term b and sigmoid transformation
}

void BaseLearner::BackPropFeatures() {
    int a, c;
    for (c = 0; c < layer1_size; c++) {
        for (a = 0; a < num_feat1; a++) {
            fea_model -> param[c * num_fea + feat_vec1[a]] += eta * part_lambda1[c];
        }
        //fea_model -> b1 -= eta * part_lambda1[c];
        fea_model -> b1s[c] += eta * part_lambda1[c];
    }
    for (c = 0; c < layer1_size; c++) {
        for (a = 0; a < num_feat2; a++) {
            fea_model -> param[c * num_fea + feat_vec2[a]] += eta * part_lambda2[c];
        }
        //fea_model -> b2 -= eta * part_lambda2[c];
        fea_model -> b2s[c] += eta * part_lambda2[c];
    }
}

void BaseLearner::TrainData(string trainfile) {
    //ofstream ofs("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/train.log");
    for (int i = 0; i < iter; i++) {
        ifstream ifs(trainfile.c_str());
        int count = 0;
        while (LoadInstance(ifs)) {
            ForwardProp();
            //ofs << ((SkipgramInstance*)inst) -> pos_scores[0] << endl;
            BackProp();
            //cout << count << ":" << fea_model -> b1s[0] << endl;
            count++;
            //if (isnan(fea_model -> b1s[0]))
//            if (count == 532)
//            {
//                cout << "here" << endl;
//            }
//            if (count == 54) {
//                cout << "here" << endl;
//            }
        }
        //eta = eta0 * (1 - i / (double)(iter + 1));
        eta = eta0;
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
    }
    printf("b1: %lf\n", fea_model -> b1s[0]);
    printf("b2: %lf\n", fea_model -> b2s[0]);
    //ofs.close();
}

void BaseLearner::BuildVocab(string trainfile) {
    word2int::iterator iter;
        ifstream ifs(trainfile.c_str());
        while (LoadInstance(ifs)) {
            string key = inst ->word1;
            iter = train_word_vocab.find(key);
            if (iter == train_word_vocab.end()) train_word_vocab[key] = 1;
            else train_word_vocab[key]++;
            key = inst ->word2;
            iter = train_word_vocab.find(key);
            if (iter == train_word_vocab.end()) train_word_vocab[key] = 1;
            else train_word_vocab[key]++;
            key = inst-> word1 + "\t" + inst ->word2;
            iter = train_phrase_vocab.find(key);
            if (iter == train_phrase_vocab.end()) train_phrase_vocab[key] = 1;
            else train_word_vocab[key]++;
        }
        ifs.close();
}

void BaseLearner::TrainData(string trainfile, string trainsubfile, string devfile) {
    int count = 0;
    eta = eta0;
    for (int i = 0; i < iter; i++) {
        ifstream ifs(trainfile.c_str());
        while (LoadInstance(ifs)) {
            ForwardProp();
            BackProp();
            count++;
            if (count % 100000 == 0) {
                cout << count << endl;
                EvalData(trainsubfile);
                EvalData(devfile);
                //eta = eta0 * (1 - count / (double)(fea_model -> num_inst + 1));
                eta = eta0;
                cout << eta << endl;
            }
        }
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
    }
    printf("b1: %lf\n", fea_model -> b1s[0]);
    printf("b2: %lf\n", fea_model -> b2s[0]);
}

void BaseLearner::SaveModel(string modelfile, bool savelm) {
    if (update_emb) {
        if (savelm) emb_model -> SaveLM(modelfile + ".emb");
        else emb_model -> SaveEmb(modelfile + ".emb");
    }
    fea_model -> SaveModel(modelfile + ".fea");
}

void BaseLearner::SaveModel(string modelfile, bool savelm, int type) {
    if (update_emb) {
        if (savelm) emb_model -> SaveLM(modelfile + ".emb", type);
        else emb_model -> SaveEmb(modelfile + ".emb");
    }
    fea_model -> SaveModel(modelfile + ".fea");
}

void BaseLearner::SaveModel(string modelfile) {
    emb_model -> SaveEmb(modelfile + ".emb");
    fea_model -> SaveModel(modelfile + ".fea");
}

void BaseLearner::LoadModel(string modelfile) {
    if (strcmp(embfile.c_str(), (modelfile+".emb").c_str()) == 0) {
        update_emb = true;
    }
    else update_emb = false;
    
    if (update_emb) {
        emb_model = new EmbeddingModel((char*)(modelfile + ".emb").c_str(), true);
    }
    else {
        emb_model = new EmbeddingModel((char*)embfile.c_str());
    }
    fea_model = new FeatureModel();
    fea_model -> LoadModel(modelfile + ".fea");
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    
    Init();
}


