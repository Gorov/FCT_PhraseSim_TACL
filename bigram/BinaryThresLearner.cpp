//
//  BinaryThresLearner.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-4.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "BinaryThresLearner.h"

void BinaryThresLearner::Init(char* embfile, char* trainfile)
{
    inst = new BinaryInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, 0);
    num_fea = fea_model -> num_fea;
    b = 0.0;
}

void BinaryThresLearner::Init(char* embfile, char* clusfile, char* trainfile)
{
    inst = new BinaryInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile, 0);
    num_fea = fea_model -> num_fea;
    b = 0.0;
}

void BinaryThresLearner::BuildVocab(string trainfile) {
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
        
        key = ((BinaryInstance*)inst) -> label;
        iter = train_word_out_vocab.find(key);
        if (iter == train_word_out_vocab.end()) train_word_out_vocab[key] = 1;
        else train_word_out_vocab[key]++;
        
        key = inst-> word1 + "\t" + inst ->word2;
        iter = train_phrase_vocab.find(key);
        if (iter == train_phrase_vocab.end()) train_phrase_vocab[key] = 1;
        else train_phrase_vocab[key]++;
    }
    ifs.close();
}

void BinaryThresLearner::EvalData(string trainfile) {
    int total = 0;
    int right = 0;
    
    int oov = 0;
    int double_oov = 0;
    int oov_err = 0;
    int double_oov_err = 0;
    int oop = 0;
    int oop_err = 0;
    int oov_out = 0;
    int oov_out_err = 0;
    
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        ForwardProp();
        if (((BinaryInstance*)inst) -> score >= 0.5 && ((BinaryInstance*)inst) -> positive == 1) {
            right++;
        }
        else if (((BinaryInstance*)inst) -> score < 0.5 && ((BinaryInstance*)inst) -> positive == 0) {
            right++;
        }
        else {
            word2int::iterator iter1, iter2;
            string key = inst ->word1;
            iter1 = train_word_vocab.find(key);
            key = inst -> word2;
            iter2 = train_word_vocab.find(key);
            
            if (iter1 == train_word_vocab.end() || iter2 == train_word_vocab.end()) oov_err++;
            if (iter1 == train_word_vocab.end() and iter2 == train_word_vocab.end()) double_oov_err++;
            
            key = inst ->word1 + "\t" + inst -> word2;
            iter1 = train_phrase_vocab.find(key);
            if (iter1 == train_phrase_vocab.end()) oop_err++;
            
            key = ((BinaryInstance*)inst) -> label;
            iter1 = train_word_out_vocab.find(key);
            if (iter1 == train_word_out_vocab.end()) oov_out_err++;
        }
        word2int::iterator iter1, iter2;
        string key = inst ->word1;
        iter1 = train_word_vocab.find(key);
        key = inst -> word2;
        iter2 = train_word_vocab.find(key);
        
        if (iter1 == train_word_vocab.end() || iter2 == train_word_vocab.end()) oov++;
        if (iter1 == train_word_vocab.end() and iter2 == train_word_vocab.end()) double_oov++;
        
        key = inst ->word1 + "\t" + inst -> word2;
        iter1 = train_phrase_vocab.find(key);
        if (iter1 == train_phrase_vocab.end()) oop++;
        
        key = ((BinaryInstance*)inst) -> label;
        iter1 = train_word_out_vocab.find(key);
        if (iter1 == train_word_out_vocab.end()) oov_out++;
        total++;
    }
//    cout << "OOV err rate: " << oov_err << " " << oov << " " << (float)oov_err / oov << endl;
//    cout << "Double OOV err rate: " << double_oov_err << " " << double_oov << " " << (float)double_oov_err / double_oov << endl;
//    cout << "OOV-out err rate: " << oov_out_err << " " << oov_out << " " << (float)oov_out_err / oov_out << endl;
//    cout << "OOP err rate: " << oop_err << " " << oop << " " << (float)oop_err / oop << endl;
    cout << "Acc: " << (float)right / total << endl;
    ifs.close();
}

void BinaryThresLearner::EvalDataObs(string trainfile) {
    int total = 0;
    int right = 0;
    
    int oov = 0;
    int double_oov = 0;
    int oov_err = 0;
    int double_oov_err = 0;
    int oop = 0;
    int oop_err = 0;
    int oov_out = 0;
    int oov_out_err = 0;
    
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        word2int::iterator iter1, iter2, iter3;
        string key = inst ->word1;
        iter1 = train_word_vocab.find(key);
        key = inst -> word2;
        iter2 = train_word_vocab.find(key);
        key = ((BinaryInstance*)inst) -> label;
        iter3 = train_word_out_vocab.find(key);
        if (iter1 == train_word_vocab.end() || iter2 == train_word_vocab.end() || iter3 == train_word_out_vocab.end()) {
            continue;
        }
        
        if (iter1 == train_word_vocab.end() || iter2 == train_word_vocab.end()) oov_err++;
        if (iter1 == train_word_vocab.end() and iter2 == train_word_vocab.end()) double_oov_err++;
        ForwardProp();
        if (((BinaryInstance*)inst) -> score >= 0.5 && ((BinaryInstance*)inst) -> positive == 1) {
            right++;
        }
        else if (((BinaryInstance*)inst) -> score < 0.5 && ((BinaryInstance*)inst) -> positive == 0) {
            right++;
        }
        total++;
    }
    cout << "Acc: " << (float)right / total << endl;
    ifs.close();
}

int BinaryThresLearner::LoadInstance(ifstream &ifs) {
    char line_buf[1000];
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    istringstream iss(line_buf);
    iss >> inst -> word1;
    iss >> inst -> tag1;
    ifs.getline(line_buf, 1000, '\n');
    iss.clear();
    iss.str(line_buf);
    iss >> inst -> word2;
    iss >> inst -> tag2;
    ifs.getline(line_buf, 1000, '\n');
    iss.clear();
    iss.str(line_buf);
    iss >> ((BinaryInstance*)inst) -> label;
    ifs.getline(line_buf, 1000, '\n');
    iss.clear();
    iss.str(line_buf);
    iss >> ((BinaryInstance*)inst) -> positive;
    
    return 1;
}

void BinaryThresLearner::ForwardOutputs()
{
    int a;
    long long l1;
    double sum;
    word2int::iterator iter;
    
    //forward output
    sum = 0.0;
    iter = emb_model -> vocabdict.find(((BinaryInstance*)inst)->label);
    if (iter == emb_model -> vocabdict.end()) {
        ((BinaryInstance*)inst) -> label_id = -1;
        ((BinaryInstance*)inst) -> score = 0.5;
    }
    else {
        ((BinaryInstance*)inst) -> label_id = iter -> second;
        l1 = iter->second * layer1_size;
        //for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn1neg[a + l1];
        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn0[a + l1];
        sum += b;
        ((BinaryInstance*)inst) -> score = Sigmoid(sum);
    }
}

long BinaryThresLearner::BackPropPhrase()
{
    int a, y;
    long long l1;
    //double sum;
    long tmpid;
    for (a = 0; a < layer1_size; a++) part_emb_p[a] = 0.0;
    
    //back phrase emb
    tmpid = ((BinaryInstance*)inst) -> label_id;
    if (tmpid < 0) {
        return -1;
    }
    //if (inst -> positive == 0) y = 1;
    //else y = 0;
    y = ((BinaryInstance*)inst)->positive;
    l1 = tmpid * layer1_size;
    
    //to check
    //for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((BinaryInstance*)inst)->score) * emb_model->syn1neg[a + l1];
    for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((BinaryInstance*)inst)->score) * emb_model->syn0[a + l1];
    if (update_emb) {
        //for (a = 0; a < layer1_size; a++) emb_model->syn1neg[a + l1] += eta * (y - ((BinaryInstance*)inst)->score) * emb_p[a];
        for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * (y - ((BinaryInstance*)inst)->score) * emb_p[a];
    }
    b += eta * (y - ((BinaryInstance*)inst)->score);
    return tmpid;
}

double BinaryThresLearner::Sigmoid(double score) {
    return 1.0 / (1 + exp(-score));
}
