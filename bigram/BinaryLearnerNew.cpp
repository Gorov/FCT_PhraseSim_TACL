//
//  BinaryLearnerNew.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-2.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "BinaryLearnerNew.h"

void BinaryLearnerNew::Init(char* embfile, char* trainfile)
{
    inst = new BinaryInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, 0);
    num_fea = fea_model -> num_fea;
}

void BinaryLearnerNew::Init(char* embfile, char* clusfile, char* trainfile)
{
    inst = new BinaryInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile, 0);
    num_fea = fea_model -> num_fea;
}

void BinaryLearnerNew::EvalData(string trainfile) {
    int total = 0;
    int right = 0;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
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

int BinaryLearnerNew::LoadInstance(ifstream &ifs) {
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

void BinaryLearnerNew::ForwardOutputs()
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
        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn1neg[a + l1];
        ((BinaryInstance*)inst) -> score = Sigmoid(sum);
    }
}

long BinaryLearnerNew::BackPropPhrase()
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
    for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((BinaryInstance*)inst)->score) * emb_model->syn1neg[a + l1];
    if (update_emb) {
        for (a = 0; a < layer1_size; a++) emb_model->syn1neg[a + l1] += eta * (y - ((BinaryInstance*)inst)->score) * emb_p[a];
    }
    return tmpid;
}

double BinaryLearnerNew::Sigmoid(double score) {
    return 1.0 / (1 + exp(-score));
}


