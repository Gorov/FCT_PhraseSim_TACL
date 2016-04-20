//
//  MRRScorer.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-14.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "MRRScorer.h"

void MRRScorer::Init() {
    inst = new PPDBInstance();
    target_inst = new PPDBTargetInstance();
}

int MRRScorer::LoadInstance(ifstream &ifs) {
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
    
    iter = emb_model -> vocabdict.find(((PPDBInstance*)inst) -> target_word1);
    if (iter != emb_model -> vocabdict.end()) ((PPDBInstance*)inst) -> target_id1 = iter -> second;
    else ((PPDBInstance*)inst) -> target_id1 = -1;
    iter = emb_model -> vocabdict.find(((PPDBInstance*)inst) -> target_word2);
    if (iter != emb_model -> vocabdict.end()) ((PPDBInstance*)inst) -> target_id2 = iter -> second;
    else ((PPDBInstance*)inst) -> target_id2 = -1;
    
    target_inst -> GetValue((PPDBInstance*)inst);
    return 1;
}

void MRRScorer::PrecomputeEmb(word2int& phrasedict, real* phraseemb) {
    //Instance inst;
    for(word2int::iterator iter = phrasedict.begin(); iter != phrasedict.end(); iter++) {
        int id = iter -> second;
        istringstream iss(iter -> first);
        iss >> inst -> word1;
        iss >> inst -> tag1;
        iss >> inst -> word2;
        iss >> inst -> tag2;
        ForwardProp();
        long long l1 = id * layer1_size;
        for (int c = 0; c < layer1_size; c++) phraseemb[c + l1] = emb_p[c];
    }
}

void MRRScorer::EvalMRR(string testfile, int threshold, string phrase_list, bool evalphrase)
{
    long long size, b, c;
    
    ifstream ifs(phrase_list.c_str());
    word2int phrasedict;
    char phraseline[1000];
    int count = 0;
    ifs.getline(phraseline, 1000, '\n');
    while (strcmp(phraseline, "") != 0) {
        string key(phraseline);
        ifs.getline(phraseline, 1000, '\n');
        key += "\t";
        key += phraseline;
        word2int::iterator iter = phrasedict.find(key);
        if (iter == phrasedict.end()) {
            phrasedict[key] = count;
            count++;
        }
        ifs.getline(phraseline, 1000, '\n');
    }
    ifs.close();
    real* phraseemb = (real*) malloc(sizeof(real) * count * layer1_size);
    PrecomputeEmb(phrasedict, phraseemb);
    
    int Total = 0;
    double score_total = 0.0;
    long long l1;
    real sim;
    
    ifs.close();
    ifs.open(testfile.c_str());
    while (LoadInstance(ifs)) {
        Total++;
        ForwardProp();
        if (inst -> id1 < 0 && inst -> id2 < 0) continue;
        //ifs.getline(phraseline, 1000, '\n');
        string key(target_inst -> word1 + '\t' + target_inst -> tag1);
        //ifs.getline(phraseline, 1000, '\n');
        key += "\t";
        key += target_inst -> word2;
        key += "\t";
        key += target_inst -> tag2;
        
        if (Total % 10 == 0) {
            printf("%d\r", Total);
            fflush(stdout);
        }
        
        int count_larger = 0;
        real t_sim = 0.0;
        
        word2int::iterator iter = phrasedict.find(key);
        if (iter == phrasedict.end()) continue;
        else {
            l1 = iter->second * layer1_size;
            for (c = 0; c < layer1_size; c++) t_sim += phraseemb[c + l1] * emb_p[c];
            //cout << t_sim << endl;
        }
        
        if (evalphrase) {
        for(word2int::iterator iter = phrasedict.begin(); iter != phrasedict.end(); iter++) {
            if (strcmp(iter -> first.c_str(), key.c_str()) != 0) {
                l1 = iter->second * layer1_size;
                sim = 0.0;
                for (c = 0; c < layer1_size; c++) sim += phraseemb[c + l1] * emb_p[c];
                if (sim > t_sim) {
                    count_larger++;
                }
            }
        }
        }
        
        for (b = 0; b < threshold; b++) {
            l1 = b * layer1_size;
            sim = 0.0;
            for (c = 0; c < layer1_size; c++) sim += (emb_model -> syn0[c + l1]) * emb_p[c];
            if (sim > t_sim) count_larger++;
        }
        
        score_total += (real)1 / (count_larger + 1);
    }
    ifs.close();
    
    vector<string> val;
    printf("\n");
    printf("MRR (threshold %d): %.2f %d %.2f %% \n", threshold, score_total, Total, score_total / Total * 100);
}

void MRRScorer::ForwardProp()
{
    int a;
    long long l1;
    
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
}

void MRRScorer::LoadModel(string modelfile) {
    emb_model = new EmbeddingModel((char*)(modelfile + ".emb").c_str(), EMB);
    fea_model = new FeatureModel();
    fea_model -> LoadModel(modelfile + ".fea");
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    
    ((BaseLearner*)(this)) -> Init();
}

void MRRScorer::LoadModel(string modelfile, bool onlyemb) {
    emb_model = new EmbeddingModel((char*)modelfile.c_str(), EMB);
    fea_model = new FeatureModel();
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    
    fea_model -> dim = layer1_size;
    fea_model -> num_fea = 0;
    
    fea_model -> b1s = (real*)malloc(layer1_size * sizeof(real));
    fea_model -> b2s = (real*)malloc(layer1_size * sizeof(real));
    
    for (int i = 0; i < fea_model -> dim; i++) fea_model -> b1s[i] = 1.0;
    for (int i = 0; i < fea_model -> dim; i++) fea_model -> b2s[i] = 1.0;
    
    ((BaseLearner*)(this)) -> Init();
}
