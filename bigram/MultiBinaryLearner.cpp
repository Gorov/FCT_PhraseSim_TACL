//
//  MultiBinaryLearner.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-2.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "MultiBinaryLearner.h"


void MultiBinaryLearner::Init(char* embfile, char* trainfile)
{
    //BaseLearner::Init(embfile, trainfile);
    inst = new Instance();
    fea_model = new FeatureModel((int)layer1_size, trainfile);
    num_fea = fea_model -> num_fea;
    five_choice = true;
}

void MultiBinaryLearner::Init(char* embfile, char* clusfile, char* trainfile)
{
    //BaseLearner::Init(embfile, trainfile);
    inst = new Instance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile);
    num_fea = fea_model -> num_fea;
    five_choice = true;
}

void MultiBinaryLearner::EvalData(string trainfile) {
    int total = 0;
    int right5 = 0;
    int right7 = 0;
    double max, max_p;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        ForwardProp();
        max = -1;
        max_p = -1;
        for (int i = 6; i >= 0; i--){
            if (((Instance*)inst) -> scores[i] > max) {
                max = ((Instance*)inst) -> scores[i];
                max_p = i;
            }
        }
        if (max_p == 0) right7++;
        max = -1;
        max_p = -1;
        for (int i = 6; i >= 0; i--){
            if (i == 1 || i == 2) continue;
            if (((Instance*)inst) -> scores[i] > max) {
                max = ((Instance*)inst) -> scores[i];
                max_p = i;
            }
        }
        if (max_p == 0) right5++;
        total++;
    }
    cout << "7-choice acc: " << (float)right7 / total << endl;
    cout << "5-choice acc: " << (float)right5 / total << endl;
    ifs.close();
}

void MultiBinaryLearner::EvalDataAcc(string trainfile) {
    int total = 0;
    int right = 0;
    double max, max_p;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        ForwardProp();
        for (int i = 6; i >= 0; i--){
            if (i == 1 || i == 2) continue;
            total ++;
            if (((Instance*)inst) -> scores[i] > 0.5 && i == 0 ) right++;
            else if (((Instance*)inst) -> scores[i] < 0.5 && i != 0 ) right++;
        }
    }
    cout << "Binary acc: " << (float)right / total << endl;
    ifs.close();
}


void MultiBinaryLearner::EvalData14(string trainfile) {
    int total = 0;
    double right14 = 0;
    double max, max_p;
    double score0 = 0;
    int equal0 = 0;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        ForwardProp();
        max = -1;
        max_p = -1;
        
        score0 = ((Instance*)inst) -> scores[0];
        for (int i = 6; i >= 0; i--){
            if (i == 1 || i == 2) continue;
            if (((Instance*)inst) -> scores[i] >= max) {
                max = ((Instance*)inst) -> scores[i];
                max_p = i;
            }
        }
        
        string tmp = inst -> word1;
        inst -> word1 = inst -> word2;
        inst -> word2 = tmp;
        
        tmp = inst -> tag1;
        inst -> tag1 = inst -> tag2;
        inst -> tag2 = tmp;
        equal0 = 1;
        ForwardProp();
        for (int i = 6; i >= 0; i--){
            if (i == 1 || i == 2) continue;
            if (((Instance*)inst) -> scores[i] > max) {
                max = ((Instance*)inst) -> scores[i];
                max_p = i + 7;
            }
            if (((Instance*)inst) -> scores[i] == score0) equal0++;
        }
        
        if (max_p == 0) right14 += 1.0/equal0;
        total++;
    }
    cout << "14-choice acc: " << (float)right14 / total << endl;
    ifs.close();
}

int MultiBinaryLearner::LoadInstance(ifstream &ifs) {
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
    
    for (int i = 0; i < 7; i++) {
        ifs.getline(line_buf, 1000, '\n');
        ((Instance*)inst) -> labels[i] = line_buf;
    }
    return 1;
}

void MultiBinaryLearner::ForwardOutputs()
{
    int a, c;
    long long l1;
    double sum;
    word2int::iterator iter;
    for (c = 0; c < ((Instance*)inst) -> labels.size(); c++) {
        sum = 0.0;
        iter = emb_model -> vocabdict.find(((Instance*)inst)->labels[c]);
        if (iter == emb_model -> vocabdict.end()) {
            ((Instance*)inst) -> ids[c] = -1;
            ((Instance*)inst) -> scores[c] = 0.5;
            continue;
        }
        ((Instance*)inst) -> ids[c] = iter -> second;
        l1 = iter -> second * layer1_size;
        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn1neg[a + l1];
        ((Instance*)inst) -> scores[c] = Sigmoid(sum);
    }
}

long MultiBinaryLearner::BackPropPhrase() {
    int a, c, y;
    long long l1;
    //double sum;
    long tmpid;
    for (a = 0; a < layer1_size; a++) part_emb_p[a] = 0.0;
    for (c = 0; c < ((Instance*)inst) -> ids.size(); c++) {
        if (five_choice) if (c == 1 || c == 2) {
            continue;
        }
        tmpid = ((Instance*)inst) -> ids[c];
        if (tmpid < 0) {
            continue;
        }
        if (c == 0) y = 1;
        else y = 0;
        l1 = tmpid * layer1_size;
        for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((Instance*)inst)->scores[c]) * emb_model->syn1neg[a + l1];
        if (update_emb) {
            for (a = 0; a < layer1_size; a++) emb_model->syn1neg[a + l1] += eta * (y - ((Instance*)inst)->scores[c]) * emb_p[a];
        }
    }
    return 0;
}

double MultiBinaryLearner::Sigmoid(double score) {
    return 1.0 / (1 + exp(-score));
}
