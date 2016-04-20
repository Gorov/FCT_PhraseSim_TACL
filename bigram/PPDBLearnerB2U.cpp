//
//  PPDBLearnerB2U.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-15.
//  Copyright (c) 2014年 hit. All rights reserved.
//


#include <iostream>
#include "PPDBLearnerB2U.h"

void PPDBLearnerB2U::Init() {
    inst = new PPDBB2UInstance();
}

void PPDBLearnerB2U::Init(char* embfile, char* trainfile)
{
    inst = new PPDBB2UInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, 2);
    num_fea = fea_model -> num_fea;
}

void PPDBLearnerB2U::Init(char* embfile, char* clusfile, char* trainfile)
{
    inst = new PPDBB2UInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile, 2);
    num_fea = fea_model -> num_fea;
    //init vocab and freq table
}

void PPDBLearnerB2U::InitWithOption(char* embfile, char* clusfile, char* trainfile)
{
    eta = eta0 = 0.01;
    iter = 1;
    
    emb_model = new EmbeddingModel(embfile, EMB);
    layer1_size = emb_model -> layer1_size;
    
    lambda1 = new real[layer1_size];
    lambda2 = new real[layer1_size];
    emb_p = new real[layer1_size];
    
    part_lambda1 = new real[layer1_size];
    part_lambda2 = new real[layer1_size];
    part_emb_p = new real[layer1_size];
    
    update_emb = true;
    five_choice = false;
    
    inst = new PPDBB2UInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile, 2);
    num_fea = fea_model -> num_fea;
}

void PPDBLearnerB2U::InitVocab(char* vocabfile) {
    next_random = 0;
    InitFreqTable(vocabfile);
    InitUnigramTable();
    
    vocab.resize(vocab_size);
    word2int::iterator iter;
    for (iter = emb_model -> vocabdict.begin(); iter != emb_model -> vocabdict.end(); iter++) {
        vocab[iter -> second] = iter -> first;
    }
}

int PPDBLearnerB2U::LoadInstance(ifstream &ifs) {
    char line_buf[1000];
    word2int::iterator iter;
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
    
    iter = emb_model -> vocabdict.find(inst->word1);
    if (iter != emb_model -> vocabdict.end()) inst -> id1 = iter -> second;
    else inst -> id1 = -1;
    iter = emb_model -> vocabdict.find(inst->word2);
    if (iter != emb_model -> vocabdict.end()) inst -> id2 = iter -> second;
    else inst -> id2 = -1;
    
    iss >> ((PPDBB2UInstance*)inst) -> pos_label;
    iter = emb_model -> vocabdict.find(((PPDBB2UInstance*)inst) -> pos_label);
    if (iter == emb_model -> vocabdict.end()) {
        ((PPDBB2UInstance*)inst) -> pos_id = -1;
    }
    else {
        ((PPDBB2UInstance*)inst) -> pos_id = iter -> second;
    }
    ((PPDBB2UInstance*)inst) -> pos_score = 0.0;
    
    for (int j = 0; j < 15; j++) {
        long id = SampleNegative();
        while (id == ((PPDBB2UInstance*)inst) -> pos_id) {
            id = SampleNegative();
        }
        string word = vocab[id];
        ((PPDBB2UInstance*)inst) -> neg_labels[j] = word;
        ((PPDBB2UInstance*)inst) -> neg_ids[j] = id;
        ((PPDBB2UInstance*)inst) -> neg_scores[j] = 0.0;
    }
    return 1;
}

void PPDBLearnerB2U::EvalData(string trainfile) {
    int total = 0;
    double right = 0;
    next_random = 0;
    int j;
    int count = 0;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        count++;
        ForwardProp();
        total++;
        for (j = 0; j < 15; j++) {
            if( ((PPDBB2UInstance*)inst) -> neg_scores[j] >= ((PPDBB2UInstance*)inst) -> pos_score) break;
        }
        if (j == 15) right += 1;
    }
    cout << right << endl;
    cout << total << endl;
    cout << "Softmax Acc: " << right / total << endl;
    ifs.close();
    //ofs.close();
}

void PPDBLearnerB2U::EvalLogLoss(string trainfile) {
    int total = 0;
    double loss = 0.0;
    next_random = 0;
    int count = 0;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        count++;
        if (inst -> id1 < 0 && inst -> id2 < 0) continue;
        ForwardProp();
        if (((PPDBB2UInstance*)inst)->pos_id < 0) continue;
        total ++;
        loss += log(((PPDBB2UInstance*)inst) -> pos_score);
    }
    cout << count << endl;
    cout << loss << endl;
    cout << total << endl;
    cout << "NCE Loss: " << loss / total << endl;
    ifs.close();
}

void PPDBLearnerB2U::EvalMRR(string testfile, int threshold)
{
    long long size, b, c;
    
    int Total = 0;
    double score_total = 0.0;
    long long l1;
    real sim;
    real norm1, norm2;
    ifstream ifs(testfile.c_str());
    while (LoadInstance(ifs)) {
        Total++;
        ForwardProp();
        if (inst -> id1 < 0 && inst -> id2 < 0) continue;
        
        if (Total % 10 == 0) {
            //printf("%d\r", Total);
            fflush(stdout);
        }
        
        int count_larger = 0;
        real t_sim = 0.0;
        
        word2int::iterator iter = emb_model -> vocabdict.find(((PPDBB2UInstance*)inst) -> pos_label );
        if (iter == emb_model -> vocabdict.end()) continue;
        else {
            l1 = iter->second * layer1_size;
            norm1 = norm2 = 0.0;
            for (c = 0; c < layer1_size; c++) norm1 += emb_model -> syn0[c + l1] * emb_model -> syn0[c + l1];
            for (c = 0; c < layer1_size; c++) norm2 += emb_p[c] * emb_p[c];
            for (c = 0; c < layer1_size; c++) t_sim += emb_model -> syn0[c + l1] * emb_p[c];
            t_sim /= (sqrt (norm1 * norm2));
        }
        
        for (b = 0; b < threshold; b++) {
            if (b == ((PPDBB2UInstance*)inst) -> pos_id) continue;
            l1 = b * layer1_size;
            sim = 0.0;
            if (b == 4000) {
                //cout << "here" << endl;
            }
            norm1 = norm2 = 0.0;
            for (c = 0; c < layer1_size; c++) norm1 += emb_model -> syn0[c + l1] * emb_model -> syn0[c + l1];
            for (c = 0; c < layer1_size; c++) norm2 += emb_p[c] * emb_p[c];
            for (c = 0; c < layer1_size; c++) sim += (emb_model -> syn0[c + l1]) * emb_p[c];
            sim /= (sqrt (norm1 * norm2));
            if (sim > t_sim) count_larger++;
        }
        
        score_total += (real)1 / (count_larger + 1);
    }
    ifs.close();
    
    vector<string> val;
    printf("\n");
    printf("MRR (threshold %d): %.2f %d %.2f %% \n", threshold, score_total, Total, score_total / Total * 100);
}

void PPDBLearnerB2U::EvalMRRDouble(string testfile, int threshold)
{
    long long size, b, c;
    
    int Total = 0;
    double score_total = 0.0;
    long long l1;
    real sim;
    
    ifstream ifs(testfile.c_str());
    while (LoadInstance(ifs)) {
        Total++;
        ForwardProp();
        if (inst -> id1 < 0 && inst -> id2 < 0) continue;
        
        if (Total % 10 == 0) {
            //printf("%d\r", Total);
            fflush(stdout);
        }
        
        int count_larger = 0;
        real t_sim = 0.0;
        real norm1, norm2;
        
        word2int::iterator iter = emb_model -> vocabdict.find(((PPDBB2UInstance*)inst) -> pos_label );
        if (iter == emb_model -> vocabdict.end()) continue;
        else {
            l1 = iter->second * layer1_size;
            norm1 = norm2 = 0.0;
            for (c = 0; c < layer1_size; c++) norm1 += emb_model -> syn1neg[c + l1] * emb_model -> syn1neg[c + l1];
            for (c = 0; c < layer1_size; c++) norm2 += emb_p[c] * emb_p[c];
            for (c = 0; c < layer1_size; c++) t_sim += emb_model -> syn1neg[c + l1] * emb_p[c];
            t_sim /= (sqrt (norm1 * norm2));
        }
        
        for (b = 0; b < threshold; b++) {
            if (b == ((PPDBB2UInstance*)inst) -> pos_id) continue;
            l1 = b * layer1_size;
            
            norm1 = norm2 = 0.0;
            for (c = 0; c < layer1_size; c++) norm1 += emb_model -> syn1neg[c + l1] * emb_model -> syn1neg[c + l1];
            for (c = 0; c < layer1_size; c++) norm2 += emb_p[c] * emb_p[c];
            
            sim = 0.0;
            for (c = 0; c < layer1_size; c++) sim += (emb_model -> syn1neg[c + l1]) * emb_p[c];
            sim /= (sqrt (norm1 * norm2));
            if (sim > t_sim) count_larger++;
        }
        
        score_total += (real)1 / (count_larger + 1);
    }
    ifs.close();
    
    vector<string> val;
    printf("\n");
    printf("MRR (threshold %d): %.2f %d %.2f %% \n", threshold, score_total, Total, score_total / Total * 100);
}

void PPDBLearnerB2U::BuildVocab(string trainfile) {
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
        
        key = ((PPDBB2UInstance*)inst) -> pos_label;
        iter = train_word_out_vocab.find(key);
        if (iter == train_word_out_vocab.end()) train_word_out_vocab[key] = 1;
        else train_word_out_vocab[key]++;
    }
    ifs.close();
}

void PPDBLearnerB2U::EvalDataObs(string trainfile) {
    int total = 0;
    double right = 0;
    next_random = 0;
    int j;
    int count = 0;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        word2int::iterator iter1, iter2, iter3;
        string key = inst ->word1;
        iter1 = train_word_vocab.find(key);
        key = inst -> word2;
        iter2 = train_word_vocab.find(key);
        key = ((PPDBB2UInstance*)inst) -> pos_label;
        iter3 = train_word_out_vocab.find(key);
        if (iter1 == train_word_vocab.end() || iter2 == train_word_vocab.end() || iter3 == train_word_out_vocab.end()) {
            continue;
        }
        count++;
        ForwardProp();
        total++;
        for (j = 0; j < 15; j++) {
            if( ((PPDBB2UInstance*)inst) -> neg_scores[j] >= ((PPDBB2UInstance*)inst) -> pos_score) break;
        }
        if (j == 15) right += 1;
    }
    cout << right << endl;
    cout << total << endl;
    cout << "Softmax Acc: " << right / total << endl;
    ifs.close();
    //ofs.close();
}

void PPDBLearnerB2U::EvalMRRObs(string testfile, int threshold)
{
    long long size, b, c;
    
    int Total = 0;
    double score_total = 0.0;
    long long l1;
    real sim;
    real norm1, norm2;
    ifstream ifs(testfile.c_str());
    while (LoadInstance(ifs)) {
        word2int::iterator iter1, iter2, iter3;
        string key = inst ->word1;
        iter1 = train_word_vocab.find(key);
        key = inst -> word2;
        iter2 = train_word_vocab.find(key);
        key = ((PPDBB2UInstance*)inst) -> pos_label;
        iter3 = train_word_out_vocab.find(key);
        if (iter1 == train_word_vocab.end() || iter2 == train_word_vocab.end() || iter3 == train_word_out_vocab.end()) {
            continue;
        }
        Total++;
        ForwardProp();
        if (inst -> id1 < 0 && inst -> id2 < 0) continue;
        
        if (Total % 10 == 0) {
            //printf("%d\r", Total);
            fflush(stdout);
        }
        
        int count_larger = 0;
        real t_sim = 0.0;
        
        word2int::iterator iter = emb_model -> vocabdict.find(((PPDBB2UInstance*)inst) -> pos_label );
        if (iter == emb_model -> vocabdict.end()) continue;
        else {
            l1 = iter->second * layer1_size;
            norm1 = norm2 = 0.0;
            for (c = 0; c < layer1_size; c++) norm1 += emb_model -> syn0[c + l1] * emb_model -> syn0[c + l1];
            for (c = 0; c < layer1_size; c++) norm2 += emb_p[c] * emb_p[c];
            for (c = 0; c < layer1_size; c++) t_sim += emb_model -> syn0[c + l1] * emb_p[c];
            t_sim /= (sqrt (norm1 * norm2));
        }
        
        for (b = 0; b < threshold; b++) {
            if (b == ((PPDBB2UInstance*)inst) -> pos_id) continue;
            l1 = b * layer1_size;
            sim = 0.0;
            if (b == 4000) {
                //cout << "here" << endl;
            }
            norm1 = norm2 = 0.0;
            for (c = 0; c < layer1_size; c++) norm1 += emb_model -> syn0[c + l1] * emb_model -> syn0[c + l1];
            for (c = 0; c < layer1_size; c++) norm2 += emb_p[c] * emb_p[c];
            for (c = 0; c < layer1_size; c++) sim += (emb_model -> syn0[c + l1]) * emb_p[c];
            sim /= (sqrt (norm1 * norm2));
            if (sim > t_sim) count_larger++;
        }
        
        score_total += (real)1 / (count_larger + 1);
    }
    ifs.close();
    
    vector<string> val;
    printf("\n");
    printf("MRR (threshold %d): %.2f %d %.2f %% \n", threshold, score_total, Total, score_total / Total * 100);
}

void PPDBLearnerB2U::ForwardOutputs()
{
    int a;
    long long l1;
    real sum;
    //positive score
    sum = 0.0;
    if (((PPDBB2UInstance*)inst)->pos_id < 0) {
        ((PPDBB2UInstance*)inst) -> pos_score = 0.0;
        return;
    }
    l1 = ((PPDBB2UInstance*)inst)->pos_id * layer1_size;
    for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn0[a + l1];
    ((PPDBB2UInstance*)inst) -> pos_score = exp(sum);
    
    //negative score
    for (int j = 0; j < 15; j++) {
        sum = 0.0;
        if (((PPDBB2UInstance*)inst)->neg_ids[j] < 0) {
            ((PPDBB2UInstance*)inst) -> neg_scores[j] = 0.0;
            continue;
        }
        l1 = ((PPDBB2UInstance*)inst)->neg_ids[j]  * layer1_size;
        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn0[a + l1];
        ((PPDBB2UInstance*)inst) -> neg_scores[j] = exp(sum);
    }
    
    //softmax
    sum = ((PPDBB2UInstance*)inst) -> pos_score;
    for (int j = 0; j < 15; j++) {
        sum += ((PPDBB2UInstance*)inst) -> neg_scores[j];
    }
    ((PPDBB2UInstance*)inst) -> pos_score /= sum;
    for (int j = 0; j < 15; j++) {
        ((PPDBB2UInstance*)inst) -> neg_scores[j] /= sum;
    }
}

long PPDBLearnerB2U::BackPropPhrase() {
    int a, y;
    long long l1;
    //double sum;
    long tmpid;
    for (a = 0; a < layer1_size; a++) part_emb_p[a] = 0.0;
    tmpid = ((PPDBB2UInstance*)inst) -> pos_id;
    if (tmpid < 0) {
        return -1;
    }
    y = 1;
    l1 = tmpid * layer1_size;
    for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((PPDBB2UInstance*)inst)->pos_score) * emb_model->syn0[a + l1];
    if (update_emb) {
        for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * (y - ((PPDBB2UInstance*)inst)->pos_score) * emb_p[a];
    }
    
    //negative bp
    for (int j = 0; j < 15; j++) {
        tmpid = ((PPDBB2UInstance*)inst) -> neg_ids[j];
        if (tmpid < 0) {
            continue;
        }
        y = 0;
        l1 = tmpid * layer1_size;
        for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((PPDBB2UInstance*)inst)->neg_scores[j]) * emb_model->syn0[a + l1];
        if (update_emb) {
            for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * (y - ((PPDBB2UInstance*)inst)->neg_scores[j]) * emb_p[a];
        }
    }
    return 0;
}

void PPDBLearnerB2U::TrainBigData(string trainfile, string trainsubfile, string devfile) {
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
                EvalLogLoss(trainsubfile);
                EvalLogLoss(devfile);
                eta = eta0 * (1 - count / (double)(fea_model -> num_inst + 1));
                //eta = eta0;
                cout << "Learning rate:" << eta << endl;
            }
        }
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
    }
    printf("b1: %lf\n", fea_model -> b1s[0]);
    printf("b2: %lf\n", fea_model -> b2s[0]);
}

void PPDBLearnerB2U::InitFreqTable(char* filename) {
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

void PPDBLearnerB2U::InitUnigramTable() {
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

long PPDBLearnerB2U::SampleNegative() {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    long ret = table[(next_random >> 16) % table_size];
    if (ret == 0) ret = next_random % (vocab_size - 1) + 1;
    //cout << next_random << " " << ret << " ";
    return ret;
}

void PPDBLearnerB2U::BuildBinaryTree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    for (a = 0; a < vocab_size; a++) count[a] = freqtable[a];
    for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
    pos1 = vocab_size - 1;
    pos2 = vocab_size;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++) {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            } else {
                min1i = pos2;
                pos2++;
            }
        } else {
            min1i = pos2;
            pos2++;
        }
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            } else {
                min2i = pos2;
                pos2++;
            }
        } else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++) {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == vocab_size * 2 - 2) break;
        }
        wordinfo_table[a].codelen = i;
        wordinfo_table[a].point[0] = (int)vocab_size - 2;
        for (b = 0; b < i; b++) {
            wordinfo_table[a].code[i - b - 1] = code[b];
            wordinfo_table[a].point[i - b] = (int)point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

void PPDBLearnerB2U::LoadModel(string modelfile) {
    emb_model = new EmbeddingModel((char*)(modelfile + ".emb").c_str(), EMB);
    fea_model = new FeatureModel();
    fea_model -> LoadModel(modelfile + ".fea");
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    
    ((BaseLearner*)(this)) -> Init();
}

void PPDBLearnerB2U::LoadModel(string modelfile, string embfile) {
    emb_model = new EmbeddingModel((char*)(modelfile + ".emb").c_str(), LM);
    //emb_model = new EmbeddingModel((char*)(modelfile).c_str(), LM);

    long long words, size, a, b;
    char ch;
    FILE *f = fopen(embfile.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    if (size != emb_model -> layer1_size) {
        cout << "inconsistant dim:" << size << emb_model -> layer1_size << endl;
    }
//    emb_model -> syn1neg = (float *)malloc(words * size * sizeof(float));
//    
//    if (emb_model -> syn1neg == NULL) {
//        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(real) / 1048576);
//        return;
//    }
    
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
        for (a = 0; a < size; a++) {
            fread(&tmp, sizeof(float), 1, f);
//            if (emb_model -> syn1neg[a + b * size] != tmp) {
//                cout << "here" << endl;
//            }
        }
    }
    fclose(f);
    
    fea_model = new FeatureModel();
    if (true) fea_model -> LoadModel(modelfile + ".fea");

    if (false) {
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

void PPDBLearnerB2U::LoadEmb(string embfile) {
    emb_model = new EmbeddingModel((char*)(embfile).c_str(), LM);
    
    fea_model = new FeatureModel();
    
        fea_model -> num_fea = 0;
        fea_model -> dim = emb_model -> layer1_size;
        fea_model -> b1s = (real*)malloc(fea_model -> dim * sizeof(real));
        fea_model -> b2s = (real*)malloc(fea_model -> dim * sizeof(real));
        for (int i = 0; i < fea_model -> dim; i++) fea_model -> b1s[i] = 1.0;
        for (int i = 0; i < fea_model -> dim; i++) fea_model -> b2s[i] = 1.0;
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    ((BaseLearner*)(this)) -> Init();
}


