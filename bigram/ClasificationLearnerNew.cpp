//
//  ClasificationLearnerNew.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-1.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "ClassificationLearnerNew.h"


void ClassificationLearnerNew::Init(char* embfile, char* trainfile)
{
    //BaseLearner::Init(embfile, trainfile);
    inst = new Instance();
    fea_model = new FeatureModel((int)layer1_size, trainfile);
    num_fea = fea_model -> num_fea;
}

void ClassificationLearnerNew::Init(char* embfile, char* clusfile, char* trainfile)
{
    //BaseLearner::Init(embfile, trainfile);
    inst = new Instance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile);
    num_fea = fea_model -> num_fea;
}

void ClassificationLearnerNew::EvalData(string trainfile) {
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


void ClassificationLearnerNew::EvalData14(string trainfile) {
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

void ClassificationLearnerNew::EvalDataDS(string trainfile) {
    int total = 0;
    int right5 = 0;
    int right7 = 0;
    double max, max_p;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        ForwardPropDS();
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


void ClassificationLearnerNew::EvalData14DS(string trainfile) {
    int total = 0;
    double right14 = 0;
    double max, max_p;
    double score0 = 0;
    int equal0 = 0;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        ForwardPropDS();
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
        ForwardPropDS();
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

int ClassificationLearnerNew::LoadInstance(ifstream &ifs) {
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

void ClassificationLearnerNew::ForwardPropDS()
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
    ForwardOutputsDS();
}

void ClassificationLearnerNew::ForwardOutputsDS()
{
    int a, c;
    long long l1;
    double sum;
    real norm1 = 1.0, norm2 = 1.0;
    word2int::iterator iter;
    for (c = 0; c < ((Instance*)inst) -> labels.size(); c++) {
        sum = 0.0;
        iter = emb_model -> vocabdict.find(((Instance*)inst)->labels[c]);
        if (iter == emb_model -> vocabdict.end()) {
            ((Instance*)inst) -> ids[c] = -1;
            ((Instance*)inst) -> scores[c] = 0.0;
            continue;
        }
        ((Instance*)inst) -> ids[c] = iter -> second;
        l1 = iter -> second * layer1_size;
        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn1neg[a + l1];
        norm1 = norm2 = 0.0;
        for (a = 0; a < layer1_size; a++) norm1 += emb_p[a] * emb_p[a];
        for (a = 0; a < layer1_size; a++) norm2 += emb_model->syn1neg[a + l1] * emb_model->syn1neg[a + l1];
        norm1 = sqrt(norm1);
        norm1 = sqrt(norm2);
        //((Instance*)inst) -> scores[c] = sum;
        ((Instance*)inst) -> scores[c] = sum / (norm1 * norm2);
    }
//    sum = 0.0;
//    for (c = 0; c < ((Instance*)inst) -> labels.size(); c++) {
//        if (five_choice) if (c == 1 || c == 2) {
//            continue;
//        }
//        ((Instance*)inst) -> scores[c] = exp(((Instance*)inst) -> scores[c]);
//        sum += ((Instance*)inst) -> scores[c];
//    }
//    for (c = 0; c < ((Instance*)inst) -> labels.size(); c++) {
//        if (five_choice) if (c == 1 || c == 2) {
//            continue;
//        }
//        ((Instance*)inst) -> scores[c] /= sum;
//    }
}

void ClassificationLearnerNew::ForwardOutputs()
{
    int a, c;
    long long l1;
    double sum;
    real norm1 = 1.0, norm2 = 1.0;
    word2int::iterator iter;
    for (c = 0; c < ((Instance*)inst) -> labels.size(); c++) {
        sum = 0.0;
        iter = emb_model -> vocabdict.find(((Instance*)inst)->labels[c]);
        if (iter == emb_model -> vocabdict.end()) {
            ((Instance*)inst) -> ids[c] = -1;
            ((Instance*)inst) -> scores[c] = 0.0;
            continue;
        }
        ((Instance*)inst) -> ids[c] = iter -> second;
        l1 = iter -> second * layer1_size;
        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn0[a + l1];
//        norm1 = norm2 = 0.0;
//        for (a = 0; a < layer1_size; a++) norm1 += emb_p[a] * emb_p[a];
//        for (a = 0; a < layer1_size; a++) norm2 += emb_model->syn0[a + l1] * emb_model->syn0[a + l1];
//        norm1 = sqrt(norm1);
//        norm1 = sqrt(norm2);
        ((Instance*)inst) -> scores[c] = sum / (norm1 * norm2);
    }
//    sum = 0.0;
//    for (c = 0; c < ((Instance*)inst) -> labels.size(); c++) {
//        if (five_choice) if (c == 1 || c == 2) {
//            continue;
//        }
//        ((Instance*)inst) -> scores[c] = exp(((Instance*)inst) -> scores[c]);
//        sum += ((Instance*)inst) -> scores[c];
//    }
//    for (c = 0; c < ((Instance*)inst) -> labels.size(); c++) {
//        if (five_choice) if (c == 1 || c == 2) {
//            continue;
//        }
//        ((Instance*)inst) -> scores[c] /= sum;
//    }
}

long ClassificationLearnerNew::BackPropPhrase() {
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

void ClassificationLearnerNew::LoadModel(string modelfile, string feafile, string embfile) {
    //emb_model = new EmbeddingModel((char*)(modelfile + ".emb").c_str(), LM);
    emb_model = new EmbeddingModel((char*)(modelfile).c_str(), LM);
    
    long long words, size, a, b;
    char ch;
    FILE *f = fopen(embfile.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    emb_model -> syn1neg = (float *)malloc(words * size * sizeof(float));
    
    if (emb_model -> syn1neg == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(real) / 1048576);
        return;
    }
    
    char tmpword[1000];
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c", tmpword, &ch);
        if (feof(f)) {
            break;
        }
        word2int::iterator iter = emb_model -> vocabdict.find(string(tmpword));
        if(iter -> second != b) {
            cout << "inconsistant vocab:" << b << tmpword << endl;
            return;
        }
        float tmp;
        for (a = 0; a < size; a++) {
            fread(&tmp, sizeof(float), 1, f);
            emb_model -> syn1neg[a + b * size] = tmp;
        }
        for (a = 0; a < size; a++) fread(&tmp, sizeof(float), 1, f);
    }
    fclose(f);
    
    fea_model = new FeatureModel();
    fea_model -> LoadModel(feafile);
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    
    ((BaseLearner*)(this)) -> Init();
}

void ClassificationLearnerNew::LoadModel(string modelfile, string feafile, int type) {
    //emb_model = new EmbeddingModel((char*)(modelfile + ".emb").c_str(), LM);
    emb_model = new EmbeddingModel((char*)(modelfile).c_str(), type);
    
    fea_model = new FeatureModel();
    if (false) fea_model -> LoadModel(feafile);
    
    if (true) {
        fea_model -> num_fea = 0;
        fea_model -> dim = emb_model -> layer1_size;
        fea_model -> b1s = (real*)malloc(fea_model -> dim * sizeof(real));
        fea_model -> b2s = (real*)malloc(fea_model -> dim * sizeof(real));
        for (int i = 0; i < fea_model -> dim; i++) fea_model -> b1s[i] = 1.0;
        for (int i = 0; i < fea_model -> dim; i++) fea_model -> b2s[i] = 1.0;
    }
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    
    ((BaseLearner*)(this)) -> Init();
}

