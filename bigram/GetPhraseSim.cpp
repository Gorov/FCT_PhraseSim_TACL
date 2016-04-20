//
//  GetPhraseSim.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-9.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <math.h>
#include <pthread.h>
#include <sstream>
#include <tr1/unordered_map>
#include "GetPhraseSim.h"

using namespace std;

const long long max_w = 500;
float* syn0;
typedef std::tr1::unordered_map<string, int > worddict;
worddict vocabdict;
long long layer1_size;

void GetPhraseSim::InitArray() {
    target_lambda1 = new real[layer1_size];
    target_lambda2 = new real[layer1_size];
    target_emb_p = new real[layer1_size];
    
    part_target_lambda1 = new real[layer1_size];
    part_target_lambda2 = new real[layer1_size];
    part_target_emb_p = new real[layer1_size];
}

void GetPhraseSim::Init() {
    inst = new PPDBInstance();
    target_inst = new PPDBTargetInstance();
    InitArray();
}

void GetPhraseSim::Init(char* embfile, char* trainfile)
{
    inst = new PPDBInstance();
    target_inst = new PPDBTargetInstance();
    InitArray();
    fea_model = new FeatureModel((int)layer1_size, trainfile, 2); //todo
    num_fea = fea_model -> num_fea;
}

void GetPhraseSim::Init(char* embfile, char* clusfile, char* trainfile)
{
    inst = new PPDBInstance();
    target_inst = new PPDBTargetInstance();
    InitArray();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile, 2); //todo
    num_fea = fea_model -> num_fea;
}

int GetPhraseSim::LoadInstance(ifstream &ifs) {
    char line_buf[1000];
    word2int::iterator iter;
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    //((SkipgramInstance*)inst) -> Clear();
    istringstream iss(line_buf);
    iss >> inst -> word1;
    iss >> inst -> tag1;
    ifs.getline(line_buf, 1000, '\n');
    iss.clear();
    iss.str(line_buf);
    iss >> inst -> word2;
    iss >> inst -> tag2;
    
    iter = emb_model -> vocabdict.find(inst->word1);
    if (iter != emb_model -> vocabdict.end()) inst -> id1 = iter -> second;
    else inst -> id1 = -1;
    iter = emb_model -> vocabdict.find(inst->word2);
    if (iter != emb_model -> vocabdict.end()) inst -> id2 = iter -> second;
    else inst -> id2 = -1;
    
    ifs.getline(line_buf, 1000, '\n');
    iss.clear();
    iss.str(line_buf);
    iss >> ((PPDBInstance*)inst) -> target_word1;
    iss >> ((PPDBInstance*)inst) -> target_tag1;
    ifs.getline(line_buf, 1000, '\n');
    iss.clear();
    iss.str(line_buf);
    iss >> ((PPDBInstance*)inst) -> target_word2;
    iss >> ((PPDBInstance*)inst) -> target_tag2;
    
    iter = emb_model -> vocabdict.find(((PPDBInstance*)inst)->target_word1);
    if (iter != emb_model -> vocabdict.end()) ((PPDBInstance*)inst) -> target_id1 = iter -> second;
    else ((PPDBInstance*)inst) -> target_id1 = -1;
    iter = emb_model -> vocabdict.find(((PPDBInstance*)inst)->target_word2);
    if (iter != emb_model -> vocabdict.end()) ((PPDBInstance*)inst) -> target_id2 = iter -> second;
    else ((PPDBInstance*)inst) -> target_id2 = -1;
    
    target_inst -> GetValue((PPDBInstance*)inst);
    return 1;
}

void GetPhraseSim::ComposeTarget() {
    int a;
    long long l1;
    
    ExtractFeatureTarget();
    ComputeLambdasTarget();
    
    for (a = 0; a < layer1_size; a++) target_emb_p[a] = 0.0;
    word2int::iterator iter = emb_model -> vocabdict.find(target_inst -> word1);
    if (iter != emb_model -> vocabdict.end()) {
        target_inst -> id1 = iter -> second;
        l1 = iter->second * layer1_size;
        for (a = 0; a < layer1_size; a++) target_emb_p[a] = target_lambda1[a] * emb_model->syn0[a + l1];
    }
    else target_inst -> id1 = -1;
    iter = emb_model -> vocabdict.find(target_inst->word2);
    if (iter != emb_model -> vocabdict.end()) {
        target_inst -> id2 = iter -> second;
        l1 = iter->second * layer1_size;
        for (a = 0; a < layer1_size; a++) target_emb_p[a] += target_lambda2[a] * emb_model->syn0[a + l1];
    }
    else target_inst -> id2 = -1;
}

void GetPhraseSim::ExtractFeatureTarget() {
    int a[2];
    fea_model -> ExtractFeaBNP(target_inst, target_feat_vec1, target_feat_vec2, a);
    target_num_feat1 = a[0];
    target_num_feat2 = a[1];
}

void GetPhraseSim::ComputeLambdasTarget() {
    int a, c;
    for (c = 0; c < layer1_size; c++) {
        target_lambda1[c] = 0.0;
        for (a = 0; a < target_num_feat1; a++) {
            target_lambda1[c] += fea_model -> param[c * num_fea + target_feat_vec1[a]];
        }
        target_lambda1[c] += fea_model -> b1s[c];
    }
    for (c = 0; c < layer1_size; c++) {
        target_lambda2[c] = 0.0;
        for (a = 0; a < target_num_feat2; a++) {
            target_lambda2[c] += fea_model -> param[c * num_fea + target_feat_vec2[a]];
        }
        target_lambda2[c] += fea_model -> b2s[c];
    }
}

void GetPhraseSim::LoadModel(string modelfile, bool unnorm) {
    emb_model = new EmbeddingModel((char*)(modelfile + ".emb").c_str(), unnorm, false);
    fea_model = new FeatureModel();
    fea_model -> LoadModel(modelfile + ".fea");
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    cout << num_fea << endl;
    ((BaseLearner*)(this)) -> Init();
}

void GetPhraseSim::LoadEmb(string modelfile, bool unnorm) {
    emb_model = new EmbeddingModel((char*)modelfile.c_str(), unnorm);
    fea_model = new FeatureModel();
    layer1_size = emb_model -> layer1_size;
    fea_model -> dim = layer1_size;
    
    int i;
    int dim = layer1_size;
    fea_model -> b1s = (real*)malloc(dim * sizeof(real));
    fea_model -> b2s = (real*)malloc(dim * sizeof(real));
    for (i = 0; i < dim; i++) fea_model -> b1s[i] = 1.0;
    for (i = 0; i < dim; i++) fea_model -> b2s[i] = 1.0;
    
    num_fea = 0;
    ((BaseLearner*)(this)) -> Init();
}

void GetPhraseSim::GetSim(string testfile, string outfile) {
    int a;
    int count1 = 0;
    int count2 = 0;
    real sim;
    real norm1;
    real norm2;
    ifstream ifs(testfile.c_str());
    ofstream ofs(outfile.c_str());
    while (LoadInstance(ifs)) {
        ForwardProp();
        ComposeTarget();
        sim = 0.0;
        count1 = 0;
        count2 = 0;
        if (inst -> id1 != -1) {
            count1++;
        }
        if (inst -> id2 != -1) {
            count1++;
        }
        if (target_inst -> id1 != -1) {
            count2++;
        }
        if (target_inst -> id2 != -1) {
            count2++;
        }
        for (a = 0; a < layer1_size; a++) emb_p[a] /= count1;
        for (a = 0; a < layer1_size; a++) target_emb_p[a] /= count2;
        for (a = 0; a < layer1_size; a++) sim += emb_p[a] * target_emb_p[a];
        //may need norms
        /*norm1 = norm2 = 0.0;
        for (a = 0; a < layer1_size; a++) norm1 += emb_p[a] * emb_p[a];
        for (a = 0; a < layer1_size; a++) norm2 += target_emb_p[a] * target_emb_p[a];
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        sim /= norm1 * norm2;
        */
        ofs << " " << sim << " ";
        ofs << "'" << inst->word1 + "_" + inst->word2 << ":";
        ofs << target_inst->word1 + "_" + target_inst->word2 << "'" << endl;
    }
    ifs.close();
    ofs.close();
}
/*
void GetPhraseSimNocomp(string testfile, string outfile) {
    ifstream ifs(testfile.c_str());
    char line[max_w * 4];
    float* neu1 = (float*) malloc(layer1_size * sizeof(float));
    float* neu2 = (float*) malloc(layer1_size * sizeof(float));
    ifs.getline(line, max_w * 4, '\n');
    int a,b;
    float sim = 0;
    while (strcmp(line, "")) {
        cout << line << "\t";
        istringstream iss(line);
        string first1; iss >> first1;
        string second1; iss >> second1;
        string first2; iss >> first2;
        string second2; iss >> second2;
        
        worddict::iterator iter11 = vocabdict.find(first1);
        worddict::iterator iter12 = vocabdict.find(second1);
        worddict::iterator iter21 = vocabdict.find(first2);
        worddict::iterator iter22 = vocabdict.find(second2);
        if ((iter11 == vocabdict.end() && iter12 == vocabdict.end())|| (iter21 == vocabdict.end() && iter22 == vocabdict.end())) {
            cout << "0.0" << endl;
            ifs.getline(line, max_w * 4, '\n');
            continue;
        }
        int count = 0;
        for (a = 0; a < layer1_size; a++) neu1[a] = 0.0;
        
        sim = 0.0;
        count = 0;
        if (iter11 != vocabdict.end() && iter21 != vocabdict.end()) {
            b = iter11 -> second;
            for (a = 0; a < layer1_size; a++) neu1[a] += syn0[a + b * layer1_size];
            b = iter21 -> second;
            for (a = 0; a < layer1_size; a++) sim += neu1[a] * syn0[a + b * layer1_size];
            count ++;
        }
        
        if (iter12 != vocabdict.end() && iter22 != vocabdict.end()) {
            b = iter12 -> second;
            for (a = 0; a < layer1_size; a++) neu1[a] += syn0[a + b * layer1_size];
            b = iter22 -> second;
            for (a = 0; a < layer1_size; a++) sim += neu1[a] * syn0[a + b * layer1_size];
            count ++;
        }
        sim /= count;
        cout << sim << endl;
        ifs.getline(line, max_w * 4, '\n');
    }
    free(neu1);
    free(neu2);
}

void GetPhraseSimHead(string testfile, string outfile) {
    ifstream ifs(testfile.c_str());
    char line[max_w * 4];
    float* neu = (float*) malloc(layer1_size * sizeof(float));
    ifs.getline(line, max_w * 4, '\n');
    int a,b;
    float sim = 0;
    while (strcmp(line, "")) {
        cout << line << "\t";
        istringstream iss(line);
        string first1; iss >> first1;
        string second1; iss >> second1;
        string first2; iss >> first2;
        string second2; iss >> second2;
        
        worddict::iterator iter = vocabdict.find(second1);
        worddict::iterator iter2 = vocabdict.find(second2);
        if (iter == vocabdict.end() || iter2 == vocabdict.end()) {
            cout << "0.0" << endl;
            ifs.getline(line, max_w * 4, '\n');
            continue;
        }
        
        for (a = 0; a < layer1_size; a++) neu[a] = 0.0;
        if (iter2 != vocabdict.end()) {
            b = iter2 -> second;
            for (a = 0; a < layer1_size; a++) neu[a] += syn0[a + b * layer1_size];
        }
        
        sim = 0.0;
        b = iter -> second;
        for (a = 0; a < layer1_size; a++) sim += neu[a] * syn0[a + b * layer1_size];
        cout << sim << endl;
        ifs.getline(line, max_w * 4, '\n');
    }
    free(neu);
}
*/
/*
 int main (int argc, const char * argv[])
 {
 if (argc < 5) {
 cout << "Usage: GetSim head|word|nocomp|phrase modelfile testfile outfile type" << endl;
 }
 LoadEmb(argv[2]);
 int len = strlen(argv[0]);
 if (strcmp(argv[0] + len - 6, "GetSim") == 0) {
 if (strcmp(argv[1], "word") == 0) {
 GetSimWord(argv[3], argv[4], argv[5]);
 }
 else if (strcmp(argv[1], "head") == 0) {
 GetSimHead(argv[3], argv[4]);
 }
 else {
 GetSimPhrase(argv[3], argv[4]);
 }
 }
 else if (strcmp(argv[0] + len - 12, "GetPhraseSim") == 0) {
 if (strcmp(argv[1], "word") == 0) {
 GetPhraseSimWord(argv[3], argv[4], argv[5]);
 }
 else if (strcmp(argv[1], "nocomp") == 0) {
 GetPhraseSimNocomp(argv[3], argv[4]);
 }
 else if (strcmp(argv[1], "head") == 0) {
 GetPhraseSimHead(argv[3], argv[4]);
 }
 else {
 GetPhraseSimPhrase(argv[3], argv[4]);
 }
 }
 return 0;
 }*/
