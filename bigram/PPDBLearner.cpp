//
//  PPDBLearner.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-8.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "PPDBLearner.h"

void PPDBLearner::InitArray() {
    target_lambda1 = new real[layer1_size];
    target_lambda2 = new real[layer1_size];
    target_emb_p = new real[layer1_size];
    
    part_target_lambda1 = new real[layer1_size];
    part_target_lambda2 = new real[layer1_size];
    part_target_emb_p = new real[layer1_size];
    
    for (int j = 0; j < NUM_NEG_INST; j++) {
        neg_target_lambda1[j] = new real[layer1_size];
        neg_target_lambda2[j] = new real[layer1_size];
        neg_target_emb_p[j] = new real[layer1_size];
        
        neg_part_target_lambda1[j] = new real[layer1_size];
        neg_part_target_lambda2[j] = new real[layer1_size];
        neg_part_target_emb_p[j] = new real[layer1_size];
    }
}

void PPDBLearner::Init() {
    inst = new PPDBInstance();
    target_inst = new PPDBTargetInstance();
    for (int j = 0; j < NUM_NEG_INST; j++) {
        neg_inst[j] = new PPDBTargetInstance();
    }
    //InitArray();
}

void PPDBLearner::Init(char* embfile, char* trainfile)
{
    inst = new PPDBInstance();
    target_inst = new PPDBTargetInstance();
    for (int j = 0; j < NUM_NEG_INST; j++) {
        neg_inst[j] = new PPDBTargetInstance();
    }
    InitArray();
    fea_model = new FeatureModel((int)layer1_size, trainfile, 2); //todo
    num_fea = fea_model -> num_fea;
}

void PPDBLearner::Init(char* embfile, char* clusfile, char* trainfile)
{
    inst = new PPDBInstance();
    target_inst = new PPDBTargetInstance();
    for (int j = 0; j < NUM_NEG_INST; j++) {
        neg_inst[j] = new PPDBTargetInstance();
    }
    InitArray();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile, 2); //todo
    num_fea = fea_model -> num_fea;
}

void PPDBLearner::InitVocab(char* vocabfile) {
    next_random = 0;
    InitFreqTable(vocabfile);
    InitUnigramTable(); //todo keep only nouns
    
    vocab.resize(vocab_size);
    word2int::iterator iter;
    for (iter = emb_model -> vocabdict.begin(); iter != emb_model -> vocabdict.end(); iter++) {
        vocab[iter -> second] = iter -> first;
    }
}

int PPDBLearner::LoadInstance(ifstream &ifs) {
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
    
    for (int j = 0; j < 15; j++) {
        long id = SampleNegative();
        string word = vocab[id];
        ((PPDBInstance*)inst) -> neg_labels[j] = word;
        ((PPDBInstance*)inst) -> neg_ids[j] = id;
        ((PPDBInstance*)inst) -> neg_scores[j] = 0.0;
    }
    
    target_inst -> GetValue((PPDBInstance*)inst);
    return 1;
}

void PPDBLearner::EvalData(string trainfile) { // todo
    int total = 0;
    double right = 0;
    next_random = 0;
    int j;
    int count = 0;
    ifstream ifs(trainfile.c_str());
    //ofstream ofs("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/test.log");
    while (LoadInstance(ifs)) {
        total++;
        ForwardProp();
        
        for (j = 0; j < NUM_NEG_INST; j++) {
            if (neg_target_scores[j] >= ((PPDBInstance*)inst) -> target_score) {
                break;
            }
        }
        if (j != NUM_NEG_INST) {
            continue;
        }
        
        for (j = 0; j < 15; j++) {
            if (((PPDBInstance*)inst) -> neg_ids[j] < 0) {
                continue;
            }
            if( ((PPDBInstance*)inst) -> neg_scores[j] >= ((PPDBInstance*)inst) -> target_score) break;
        }
        
        if (j == 15) right += 1;
//        else {
//            cout << "here" << endl;
//        }
    }
    cout << right << endl;
    cout << total << endl;
    cout << "Softmax Acc: " << right / total << endl;
    ifs.close();
    //ofs.close();
}

void PPDBLearner::EvalLogLoss(string trainfile) {
    int total = 0;
    double loss = 0.0;
    next_random = 0;
    int j;
    int count = 0;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        total++;
        if (inst -> id1 < 0 && inst -> id2 < 0) continue;
        ForwardProp();
        if (target_inst -> id1 < 0 && target_inst -> id2 < 0) continue;
        loss += log(((PPDBInstance*)inst) -> target_score);
    }
    cout << loss << endl;
    cout << total << endl;
    cout << "NCE Loss: " << loss / total << endl;
    ifs.close();
}

void PPDBLearner::ComposeTarget() {
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
    
    for (int j = 0; j < NUM_NEG_INST; j++) {
        for (a = 0; a < layer1_size; a++) neg_target_emb_p[j][a] = 0.0;
        word2int::iterator iter = emb_model -> vocabdict.find(neg_inst[j] -> word1);
        if (iter != emb_model -> vocabdict.end()) {
            neg_inst[j] -> id1 = iter -> second;
            l1 = iter->second * layer1_size;
            for (a = 0; a < layer1_size; a++) neg_target_emb_p[j][a] = neg_target_lambda1[j][a] * emb_model->syn0[a + l1];
        }
        else neg_inst[j] -> id1 = -1;
        iter = emb_model -> vocabdict.find(neg_inst[j]->word2);
        if (iter != emb_model -> vocabdict.end()) {
            neg_inst[j] -> id2 = iter -> second;
            l1 = iter->second * layer1_size;
            for (a = 0; a < layer1_size; a++) neg_target_emb_p[j][a] += neg_target_lambda2[j][a] * emb_model->syn0[a + l1];
        }
        else neg_inst[j] -> id2 = -1;
    }
}

void PPDBLearner::ExtractFeatureTarget() {
    int a[2];
    fea_model -> ExtractFeaBNP(target_inst, target_feat_vec1, target_feat_vec2, a);
    target_num_feat1 = a[0];
    target_num_feat2 = a[1];
    for (int j = 0; j < NUM_NEG_INST; j++) {
        fea_model -> ExtractFeaBNP(neg_inst[j], neg_target_feat_vec1[j], neg_target_feat_vec2[j], a);
        neg_target_num_feat1[j] = a[0];
        neg_target_num_feat2[j] = a[1];
    }
}

void PPDBLearner::ComputeLambdasTarget() {
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
    for (int j = 0; j < NUM_NEG_INST; j++) {
        for (c = 0; c < layer1_size; c++) {
            neg_target_lambda1[j][c] = 0.0;
            for (a = 0; a < neg_target_num_feat1[j]; a++) {
                neg_target_lambda1[j][c] += fea_model -> param[c * num_fea + neg_target_feat_vec1[j][a]];
            }
            neg_target_lambda1[j][c] += fea_model -> b1s[c];
        }
        for (c = 0; c < layer1_size; c++) {
            neg_target_lambda2[j][c] = 0.0;
            for (a = 0; a < neg_target_num_feat2[j]; a++) {
                neg_target_lambda2[j][c] += fea_model -> param[c * num_fea + neg_target_feat_vec2[j][a]];
            }
            neg_target_lambda2[j][c] += fea_model -> b2s[c];
        }
    }
}

void PPDBLearner::GetNegInsts() {
    neg_inst[0] -> word1 = target_inst -> word2;
    neg_inst[0] -> word2 = target_inst -> word1;
    neg_inst[0] -> tag1 = target_inst -> tag2;
    neg_inst[0] -> tag2 = target_inst -> tag1;
    neg_inst[0] -> id1 = target_inst -> id2;
    neg_inst[0] -> id2 = target_inst -> id1;
}

void PPDBLearner::ForwardOutputs() {
    int a, c;
    long long l1;
    real sum;
    
    GetNegInsts();
    ComposeTarget();
    
    //positive score
    sum = 0.0;
    if (target_inst -> id1 < 0 && target_inst -> id2 < 0) {
        ((PPDBInstance*)inst) -> target_score = 0.0;
        return;
    }
    for (a = 0; a < layer1_size; a++) sum += emb_p[a] * target_emb_p[a];
    //cout << sum << endl;
    if (sum < MAX_EXP && sum > -MAX_EXP) ((PPDBInstance*)inst) -> target_score = exp(sum);
    else if (sum > MAX_EXP) ((PPDBInstance*)inst) -> target_score = 1e36;
    else if (sum < -MAX_EXP) ((PPDBInstance*)inst) -> target_score = 1e-36;
    
    //negative score
    for (int j = 0; j < NUM_NEG_INST; j++) {
        sum = 0.0;
        if (neg_inst[j] -> id1 < 0 && neg_inst[j] -> id2 < 0) {
            neg_target_scores[j] = 0.0;
            continue;
        }
        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * neg_target_emb_p[j][a];
        if (sum < MAX_EXP && sum > -MAX_EXP) neg_target_scores[j] = exp(sum);
        else if (sum > MAX_EXP) neg_target_scores[j] = 1e36;
        else if (sum < -MAX_EXP) neg_target_scores[j] = 1e-36;
        //neg_target_scores[j] = exp(sum);
    }
    
    for (int j = 0; j < 15; j++) {
        sum = 0.0;
        if (((PPDBInstance*)inst)->neg_ids[j] < 0) {
            ((PPDBInstance*)inst) -> neg_scores[j] = 0.0;
            continue;
        }
        l1 = ((PPDBInstance*)inst)->neg_ids[j]  * layer1_size;
        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn0[a + l1];
        //((PPDBInstance*)inst) -> neg_scores[j] = exp(sum);
        if (sum < MAX_EXP && sum > -MAX_EXP) {
            ((PPDBInstance*)inst) -> neg_scores[j] = exp(sum);
        }
        else if (sum > MAX_EXP) ((PPDBInstance*)inst) -> neg_scores[j] = 1e36;
        else if (sum < -MAX_EXP) ((PPDBInstance*)inst) -> neg_scores[j] = 1e-36;
    }
    
    //softmax
    sum = ((PPDBInstance*)inst) -> target_score;
    for (int j = 0; j < NUM_NEG_INST; j++) {
        sum += neg_target_scores[j];
    }
    for (int j = 0; j < 15; j++) {
        sum += ((PPDBInstance*)inst) -> neg_scores[j];
    }
    ((PPDBInstance*)inst) -> target_score /= sum;
    for (int j = 0; j < NUM_NEG_INST; j++) {
        neg_target_scores[j] /= sum;
    }
    for (int j = 0; j < 15; j++) {
        ((PPDBInstance*)inst) -> neg_scores[j] /= sum;
    }
}

long PPDBLearner::BackPropPhrase() {
    int a, c, y;
    long long l1;
    //double sum;
    long tmpid;
    for (a = 0; a < layer1_size; a++) part_emb_p[a] = 0.0;
    
    if (target_inst -> id1 < 0 && target_inst -> id2 < 0) {
        ((PPDBInstance*)inst) -> target_score = 0.0;
        return -1;
    }
    y = 1;
    for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((PPDBInstance*)inst)->target_score) * target_emb_p[a];
    for (a = 0; a < layer1_size; a++) part_target_emb_p[a] = 0.0;
    for (a = 0; a < layer1_size; a++) part_target_emb_p[a] += (y - ((PPDBInstance*)inst)->target_score) * emb_p[a];
    y = 0;
    for (int j = 0; j < NUM_NEG_INST; j++) {
        for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - neg_target_scores[j]) * neg_target_emb_p[j][a];
        for (a = 0; a < layer1_size; a++) neg_part_target_emb_p[j][a] = 0.0;
        for (a = 0; a < layer1_size; a++) neg_part_target_emb_p[j][a] += (y - neg_target_scores[j]) * emb_p[a];
    }
    
    BackPropTarget();

    for (int j = 0; j < 15; j++) {
        tmpid = ((PPDBInstance*)inst) -> neg_ids[j];
        if (tmpid < 0) {
            continue;
        }
        y = 0;
        l1 = tmpid * layer1_size;
        for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((PPDBInstance*)inst)->neg_scores[j]) * emb_model->syn0[a + l1];
        if (update_emb) {
            for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * (y - ((PPDBInstance*)inst)->neg_scores[j]) * emb_p[a];
        }
    }
    return 0;
}

void PPDBLearner::BackPropTarget() {
    int a;
    long long l1;
    for (a = 0; a < layer1_size; a++) part_target_lambda1[a] = 0.0;
    for (a = 0; a < layer1_size; a++) part_target_lambda2[a] = 0.0;
    if (target_inst -> id1 >= 0) {
        l1 = target_inst -> id1 * layer1_size;
        for (a = 0; a < layer1_size; a++) {
            part_target_lambda1[a] = part_target_emb_p[a] * emb_model->syn0[a + l1];
        }
        if (update_emb) for (a = 0; a < layer1_size; a++) {
            emb_model->syn0[a + l1] += eta * part_target_emb_p[a] * target_lambda1[a];
        }
    }
    if (target_inst -> id2 >= 0) {
        l1 = target_inst -> id2 * layer1_size;
        for (a = 0; a < layer1_size; a++) {
            part_target_lambda2[a] = part_target_emb_p[a] * emb_model->syn0[a + l1];
        }
        if (update_emb) for (a = 0; a < layer1_size; a++) {
            emb_model->syn0[a + l1] += eta * part_target_emb_p[a] * target_lambda2[a];
        }
    }
    for (int j = 0; j < NUM_NEG_INST; j++) {
        for (a = 0; a < layer1_size; a++) neg_part_target_lambda1[j][a] = 0.0;
        for (a = 0; a < layer1_size; a++) neg_part_target_lambda2[j][a] = 0.0;
        if (neg_inst[j] -> id1 >= 0) {
            l1 = neg_inst[j] -> id1 * layer1_size;
            for (a = 0; a < layer1_size; a++) {
                neg_part_target_lambda1[j][a] = neg_part_target_emb_p[j][a] * emb_model->syn0[a + l1];
            }
            if (update_emb) for (a = 0; a < layer1_size; a++) {
                emb_model->syn0[a + l1] += eta * neg_part_target_emb_p[j][a] * neg_target_lambda1[j][a];
            }
        }
        if (neg_inst[j] -> id2 >= 0) {
            l1 = neg_inst[j] -> id2 * layer1_size;
            for (a = 0; a < layer1_size; a++) {
                neg_part_target_lambda2[j][a] = neg_part_target_emb_p[j][a] * emb_model->syn0[a + l1];
            }
            if (update_emb) for (a = 0; a < layer1_size; a++) {
                emb_model->syn0[a + l1] += eta * neg_part_target_emb_p[j][a] * neg_target_lambda2[j][a];
            }
        }
    }
    BackPropFeaturesTarget();
}

void PPDBLearner::BackPropFeaturesTarget() {
    int a, c;
    for (c = 0; c < layer1_size; c++) {
        for (a = 0; a < target_num_feat1; a++) {
            fea_model -> param[c * num_fea + target_feat_vec1[a]] += eta * part_target_lambda1[c];
        }
        fea_model -> b1s[c] += eta * part_target_lambda1[c];
    }
    for (c = 0; c < layer1_size; c++) {
        for (a = 0; a < target_num_feat2; a++) {
            fea_model -> param[c * num_fea + target_feat_vec2[a]] += eta * part_target_lambda2[c];
        }
        fea_model -> b2s[c] += eta * part_target_lambda2[c];
    }
    for (int j = 0; j < NUM_NEG_INST; j++) {
        for (c = 0; c < layer1_size; c++) {
            for (a = 0; a < neg_target_num_feat1[j]; a++) {
                fea_model -> param[c * num_fea + neg_target_feat_vec1[j][a]] += eta * neg_part_target_lambda1[j][c];
            }
            fea_model -> b1s[c] += eta * neg_part_target_lambda1[j][c];
        }
        for (c = 0; c < layer1_size; c++) {
            for (a = 0; a < neg_target_num_feat2[j]; a++) {
                fea_model -> param[c * num_fea + neg_target_feat_vec2[j][a]] += eta * neg_part_target_lambda2[j][c];
            }
            fea_model -> b2s[c] += eta * neg_part_target_lambda2[j][c];
        }
    }
}

void PPDBLearner::TrainBigData(string trainfile, string trainsubfile, string devfile) {
    int count = 0;
    eta = eta0;
    for (int i = 0; i < iter; i++) {
        ifstream ifs(trainfile.c_str());
        while (LoadInstance(ifs)) {
            ForwardProp();
            BackProp();
            count++;
            if (count % 50000 == 0) {
                cout << count << endl;
                EvalLogLoss(trainsubfile);
                EvalLogLoss(devfile);
                eta = eta0 * (1 - count / (double)(fea_model -> num_inst + 1));
                if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
                //eta = eta0;
                cout << "Learning rate:" << eta << endl;
            }
        }
        ifs.close();
    }
    printf("b1: %lf\n", fea_model -> b1s[0]);
    printf("b2: %lf\n", fea_model -> b2s[0]);
}

void PPDBLearner::InitFreqTable(char* filename) {
    char line_buf[1000];
    string word;
    int id;
    int freq;
    freqtable = (int*)malloc(sizeof(int) * emb_model -> vocabdict.size());
    for (int i = 0; i < emb_model -> vocabdict.size(); i++) freqtable[i] = 0;
    word2int::iterator iter;
    ifstream ifs(filename);
    ifs.getline(line_buf, 1000, '\n');
    while (strcmp(line_buf, "") != 0) {
        istringstream iss(line_buf);
        iss >> word;
        iss >> freq;
        iter = emb_model -> vocabdict.find(word);
        if (iter != emb_model -> vocabdict.end()) {
            id = iter -> second;
            freqtable[id] = freq;
            //cout << id << " " << freq << endl;
        }
        ifs.getline(line_buf, 1000, '\n');    
    }
    ifs.close();
}

void PPDBLearner::InitUnigramTable() {
    int a, i;
    long long train_words_pow = 0;
    //double train_words_pow = 0.0;
    //real d1, power = 0.75;
    double d1, power = 0.75;
    //double power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    vocab_size = emb_model -> vocab_size;
    for (a = 0; a < vocab_size ; a++) {
        //cout << a << " ";
        //printf("%20.5f ", train_words_pow + pow(freqtable[a], power));
        train_words_pow += pow(freqtable[a], power);
        //cout << " " << train_words_pow << " " << freqtable[a] << " ";
        //printf("%20.5f\n", pow(freqtable[a], power));
    }
    //for (a = 0; a < vocab_size ; a++) train_words_pow += pow((double)freqtable[a], (double)power);
    cout << sizeof(long long) << endl;
    cout << sizeof(double) << endl;
    cout << sizeof(float) << endl;
    cout << train_words_pow << endl;
    i = 0;
    d1 = pow(freqtable[i], power) / (real)train_words_pow;
    for (a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (real)table_size > d1) {
            i++;
            d1 += pow(freqtable[i], power) / (real)train_words_pow;
            //cout << i << " " << a << endl;
        }
        if (i >= vocab_size) i = (int)(vocab_size - 1);
    }
}

long PPDBLearner::SampleNegative() {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    long ret = table[(next_random >> 16) % table_size];
    if (ret == 0) ret = next_random % (vocab_size - 1) + 1;
    //cout << next_random << " " << ret << " ";
    return ret;
}

void PPDBLearner::LoadModel(string modelfile) {
    emb_model = new EmbeddingModel((char*)(modelfile + ".emb").c_str(), EMB);
    fea_model = new FeatureModel();
    fea_model -> LoadModel(modelfile + ".fea");
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    
    ((BaseLearner*)(this)) -> Init();
    InitArray();
    
}
