//
//  SkipgramLearner.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-2.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "SkipgramLearner.h"

void SkipgramLearner::Init() {
    inst = new SkipgramInstance();
}

void SkipgramLearner::Init(char* embfile, char* trainfile)
{
    inst = new SkipgramInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, 2);
    num_fea = fea_model -> num_fea;
}

void SkipgramLearner::Init(char* embfile, char* clusfile, char* trainfile)
{
    inst = new SkipgramInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile, 2);
    num_fea = fea_model -> num_fea;
    //init vocab and freq table
}

void SkipgramLearner::InitWithOption(char* embfile, char* clusfile, char* trainfile)
{
    eta = eta0 = 0.01;
    iter = 1;
    
    emb_model = new EmbeddingModel(embfile, unnorm, halflm);
    layer1_size = emb_model -> layer1_size;
    
    lambda1 = new real[layer1_size];
    lambda2 = new real[layer1_size];
    emb_p = new real[layer1_size];
    
    part_lambda1 = new real[layer1_size];
    part_lambda2 = new real[layer1_size];
    part_emb_p = new real[layer1_size];
    
    update_emb = true;
    five_choice = false;
    
    inst = new SkipgramInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile, 2);
    num_fea = fea_model -> num_fea;
}

void SkipgramLearner::InitVocab(char* vocabfile) {
    next_random = 0;
    InitFreqTable(vocabfile);
    InitUnigramTable();
    
    vocab.resize(vocab_size);
    word2int::iterator iter;
    for (iter = emb_model -> vocabdict.begin(); iter != emb_model -> vocabdict.end(); iter++) {
        vocab[iter -> second] = iter -> first;
    }
}

int SkipgramLearner::LoadInstance(ifstream &ifs) {
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
    ifs.getline(line_buf, 1000, '\n');
    iss.clear();
    iss.str(line_buf);
    
    iter = emb_model -> vocabdict.find(inst->word1);
    if (iter != emb_model -> vocabdict.end()) inst -> id1 = iter -> second;
    else inst -> id1 = -1;
    iter = emb_model -> vocabdict.find(inst->word2);
    if (iter != emb_model -> vocabdict.end()) inst -> id2 = iter -> second;
    else inst -> id2 = -1;
     
    iss >> ((SkipgramInstance*)inst) -> num;
    for (int i = 0; i < ((SkipgramInstance*)inst) -> num; i++) {
        iss >> ((SkipgramInstance*)inst) -> pos_labels[i];
        iter = emb_model -> vocabdict.find(((SkipgramInstance*)inst) -> pos_labels[i]);
        if (iter == emb_model -> vocabdict.end()) {
            ((SkipgramInstance*)inst) -> pos_ids[i] = -1;
        }
        else {
            ((SkipgramInstance*)inst) -> pos_ids[i] = iter -> second;
        }
        ((SkipgramInstance*)inst) -> pos_scores[i] = 0.0;
        
        for (int j = 0; j < 15; j++) {
            long id = SampleNegative();
            while (id == ((SkipgramInstance*)inst) -> pos_ids[i]) {
                id = SampleNegative();
            }
            string word = vocab[id];
            ((SkipgramInstance*)inst) -> neg_labels[i * 15 + j] = word;
            ((SkipgramInstance*)inst) -> neg_ids[i * 15 + j] = id;
            ((SkipgramInstance*)inst) -> neg_scores[i * 15 + j] = 0.0;
        }
    }
    return 1;
}

void SkipgramLearner::EvalData(string trainfile) {
    int total = 0;
    double right = 0;
    next_random = 0;
    int j;
    int count = 0;
    ifstream ifs(trainfile.c_str());
    //ofstream ofs("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/test.log");
    while (LoadInstance(ifs)) {
        count++;
        //if (count == 795) {
        //    cout << "here" << endl;
        //}
        ForwardProp();
        for (int c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
            total ++;
            //cout << ((SkipgramInstance*)inst) -> pos_scores[c];
            for (j = 0; j < 15; j++) {
                //cout << " " << ((SkipgramInstance*)inst) -> neg_ids[c * 15 + j];
                //cout << " " << ((SkipgramInstance*)inst) -> neg_scores[c * 15 + j];
                if( ((SkipgramInstance*)inst) -> neg_scores[c * 15 + j] >= ((SkipgramInstance*)inst) -> pos_scores[c] ) break;
            }
            //cout << endl;
            //if (total == 6874) {//22280
            //    cout << "here" << endl;
            //}
            if (j == 15) right += 1;
            //else ofs << total << " " << ((SkipgramInstance*)inst) -> pos_scores[c] << " " << ((SkipgramInstance*)inst) -> neg_ids[c * 15 + j] << endl;
        }
    }
    cout << right << endl;
    cout << total << endl;
    cout << "Softmax Acc: " << right / total << endl;
    ifs.close();
    //ofs.close();
}

void SkipgramLearner::EvalLogLoss(string trainfile) {
    int total = 0;
    double loss = 0.0;
    next_random = 0;
    int count = 0;
    real tmploss = 0.0;
    int id, d;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        count++;
        if (inst -> id1 < 0 && inst -> id2 < 0) continue;
        ForwardProp();
        if (negative) for (int c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
            if (((SkipgramInstance*)inst)->pos_ids[c] < 0) continue;
            total ++;
            loss += log(((SkipgramInstance*)inst) -> pos_scores[c]);
        }
        if (hs) {
            for (int c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
                if (((SkipgramInstance*)inst)->pos_ids[c] < 0) continue;
                total ++;
                id = ((SkipgramInstance*)inst)->pos_ids[c];
                
                for (d = 0; d < wordinfo_table[id].codelen; d++) {
                    if (wordinfo_table[id].code[d] == 0) loss += log(1 / (1 + exp(-scores[c * 40 + d])));
                    else loss += log(1 / (1 + exp(scores[c * 40 + d])));
                    //loss += log(1 / (1 + exp(-scores[c * 40 + d])));
                }
            }
        }
    }
    //cout << count << endl;
    //cout << loss << endl;
    //cout << total << endl;
    cout << "NCE Loss: " << loss / total << endl;
    ifs.close();
}

void SkipgramLearner::EvalLogLossBase(string trainfile) {
    int total2 = 0;
    double loss2 = 0;
    next_random = 0;
    int count = 0;
    int id, d;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        count++;
        if (inst -> id1 >= 0) {
            ForwardBaselineLeft();
            if (negative) for (int c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
                if (((SkipgramInstance*)inst)->pos_ids[c] < 0) continue;
                total2 ++;
                loss2 += log(((SkipgramInstance*)inst) -> pos_scores[c]);
            }
            if (hs) for (int c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
                if (((SkipgramInstance*)inst)->pos_ids[c] < 0) continue;
                total2 ++;
                id = ((SkipgramInstance*)inst)->pos_ids[c];
                    
                for (d = 0; d < wordinfo_table[id].codelen; d++) {
                    if (wordinfo_table[id].code[d] == 0) loss2 += log(1 / (1 + exp(-scores[c * 40 + d])));
                    else loss2 += log(1 / (1 + exp(scores[c * 40 + d])));
                }
            }
        }
        if (inst -> id2 >= 0) {
            ForwardBaselineRight();
            if (negative) for (int c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
                if (((SkipgramInstance*)inst)->pos_ids[c] < 0) continue;
                total2 ++;
                loss2 += log(((SkipgramInstance*)inst) -> pos_scores[c]);
            }
            if (hs) for (int c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
                if (((SkipgramInstance*)inst)->pos_ids[c] < 0) continue;
                total2 ++;
                id = ((SkipgramInstance*)inst)->pos_ids[c];
                
                for (d = 0; d < wordinfo_table[id].codelen; d++) {
                    if (wordinfo_table[id].code[d] == 0) loss2 += log(1 / (1 + exp(-scores[c * 40 + d])));
                    else loss2 += log(1 / (1 + exp(scores[c * 40 + d])));
                }
            }        }
    }
    cout << count << endl;
    cout << loss2 << endl;
    cout << total2 << endl;
    cout << "Base NCE Loss: " << loss2 / total2 << endl;
    ifs.close();
}

void SkipgramLearner::ForwardBaselineLeft()
{
    int a;
    long long l1;
    
    for (a = 0; a < layer1_size; a++) emb_p[a] = 0.0;
    word2int::iterator iter = emb_model -> vocabdict.find(inst->word1);
    if (iter != emb_model -> vocabdict.end()) {
        inst -> id1 = iter -> second;
        l1 = iter->second * layer1_size;
        for (a = 0; a < layer1_size; a++) emb_p[a] = emb_model->syn0[a + l1];
    }
    else inst -> id1 = -1;
    ForwardOutputs();
}

void SkipgramLearner::ForwardBaselineRight()
{
    int a;
    long long l1;
    
    for (a = 0; a < layer1_size; a++) emb_p[a] = 0.0;
    word2int::iterator iter = emb_model -> vocabdict.find(inst->word2);
    if (iter != emb_model -> vocabdict.end()) {
        inst -> id2 = iter -> second;
        l1 = iter->second * layer1_size;
        for (a = 0; a < layer1_size; a++) emb_p[a] = emb_model->syn0[a + l1];
    }
    else inst -> id2 = -1;
    ForwardOutputs();
}

void SkipgramLearner::ForwardOutputs()
{
    int a, c, d, id;
    long long l1, l2;
    real sum;
    //word2int::iterator iter;
    for (c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
        if (((SkipgramInstance*)inst)->pos_ids[c] < 0) {
            ((SkipgramInstance*)inst) -> pos_scores[c] = 0.0;
            continue;
        }
        id = (int)((SkipgramInstance*)inst)->pos_ids[c];
        if (hs) for (d = 0; d < wordinfo_table[id].codelen; d++) {
            sum = 0.0;
            l2 = wordinfo_table[id].point[d] * layer1_size;
            // Propagate hidden -> output
            for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model -> syn1[a + l2];
            scores[c * 40 + d] = sum;
        }
        if (negative) {
            //positive score
            sum = 0.0;
            if (((SkipgramInstance*)inst)->pos_ids[c] < 0) {
                ((SkipgramInstance*)inst) -> pos_scores[c] = 0.0;
                continue;
            }
            l1 = ((SkipgramInstance*)inst)->pos_ids[c] * layer1_size;
            for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn1neg[a + l1];
            ((SkipgramInstance*)inst) -> pos_scores[c] = exp(sum);
            
            //negative score
            for (int j = 0; j < 15; j++) {
                sum = 0.0;
                if (((SkipgramInstance*)inst)->neg_ids[c * 15 + j] < 0) {
                    ((SkipgramInstance*)inst) -> neg_scores[c * 15 + j] = 0.0;
                    continue;
                }
                l1 = ((SkipgramInstance*)inst)->neg_ids[c * 15 + j]  * layer1_size;
                for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn1neg[a + l1];
                //((SkipgramInstance*)inst) -> neg_scores[c * 15 + j] = sum;
                ((SkipgramInstance*)inst) -> neg_scores[c * 15 + j] = exp(sum);
            }
            
            //softmax
            sum = ((SkipgramInstance*)inst) -> pos_scores[c];
            for (int j = 0; j < 15; j++) {
                sum += ((SkipgramInstance*)inst) -> neg_scores[c * 15 + j];
            }
            ((SkipgramInstance*)inst) -> pos_scores[c] /= sum;
            for (int j = 0; j < 15; j++) {
                ((SkipgramInstance*)inst) -> neg_scores[c * 15 + j] /= sum;
            }
        }
    }
}

long SkipgramLearner::BackPropPhrase() {
    int a, c, d, y;
    long long l1, l2;
    //double sum;
    long tmpid;
    real g;
    for (a = 0; a < layer1_size; a++) part_emb_p[a] = 0.0;
    
    for (c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
        tmpid = ((SkipgramInstance*)inst) -> pos_ids[c];
        if (tmpid < 0) {
            continue;
        }
        if (hs) for (d = 0; d < wordinfo_table[tmpid].codelen; d++) {
            if (scores[c * 40 + d] <= -max_exp) scores[c * 40 + d] = 0;
            else if (scores[c * 40 + d] >= max_exp) scores[c * 40 + d] = 1;
            else scores[c * 40 + d] = expTable[(int)((scores[c * 40 + d] + max_exp) * (exp_table_size / max_exp / 2))];
            // 'g' is the gradient multiplied by the learning rate
            g = (1 - wordinfo_table[tmpid].code[d] - scores[c * 40 + d]) * eta;
            l2 = wordinfo_table[tmpid].point[d] * layer1_size;
            // Propagate errors output -> hidden
            for (a = 0; a < layer1_size; a++) part_emb_p[a] += g * emb_model -> syn1[a + l2];
            // Learn weights hidden -> output
            for (a = 0; a < layer1_size; a++) emb_model -> syn1[a + l2] += g * emb_p[a];
        }
        if (negative) {
            y = 1;
            l1 = tmpid * layer1_size;
            for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((SkipgramInstance*)inst)->pos_scores[c]) * emb_model->syn1neg[a + l1];
            if (update_emb) {
                for (a = 0; a < layer1_size; a++) emb_model->syn1neg[a + l1] += eta * (y - ((SkipgramInstance*)inst)->pos_scores[c]) * emb_p[a];
            }
            
            //negative bp
            for (int j = 0; j < 15; j++) {
                tmpid = ((SkipgramInstance*)inst) -> neg_ids[c * 15 + j];
                if (tmpid < 0) {
                    continue;
                }
                y = 0;
                l1 = tmpid * layer1_size;
                for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((SkipgramInstance*)inst)->neg_scores[c * 15 + j]) * emb_model->syn1neg[a + l1];
                if (update_emb) {
                    for (a = 0; a < layer1_size; a++) emb_model->syn1neg[a + l1] += eta * (y - ((SkipgramInstance*)inst)->neg_scores[c * 15 + j]) * emb_p[a];
                }
            }
        }
    }
    return 0;
}

void SkipgramLearner::TrainBigData(string trainfile, string trainsubfile, string devfile) {
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

void SkipgramLearner::InitFreqTable(char* filename) {
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

void SkipgramLearner::InitUnigramTable() {
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
    //cout << sizeof(long long) << endl;
    //cout << sizeof(double) << endl;
    //cout << sizeof(float) << endl;
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

long SkipgramLearner::SampleNegative() {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    long ret = table[(next_random >> 16) % table_size];
    if (ret == 0) ret = next_random % (vocab_size - 1) + 1;
    //cout << next_random << " " << ret << " ";
    return ret;
}

void SkipgramLearner::BuildBinaryTree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    wordinfo_table = (struct word_info *)calloc((vocab_size + 1), sizeof(struct word_info));
    for (a = 0; a < vocab_size; a++) count[a] = freqtable[a];
    count[0] = 50000;
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
    
    FILE* fileout = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.vocab3", "w");
    for (a = 0; a < vocab_size; a++) {
        fprintf(fileout, "%s", vocab[a].c_str());
        for (int d = 0; d < wordinfo_table[a].codelen; d++){
            fprintf(fileout, " %d:%d", wordinfo_table[a].point[d], wordinfo_table[a].code[d]);
        }
        fprintf(fileout, "\n");
    }
    fclose(fileout);
}

void SkipgramLearner::InitExpTable() {
    int i;
    expTable = (real *)malloc((exp_table_size + 1) * sizeof(real));
    for (i = 0; i < exp_table_size; i++) {
        expTable[i] = exp((i / (real)exp_table_size * 2 - 1) * max_exp); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
}

void SkipgramLearner::LoadModel(string modelfile, bool loadlm) {
    if (loadlm) {
        emb_model = new EmbeddingModel((char*)(modelfile + ".emb").c_str(), unnorm, halflm);
    }
    fea_model = new FeatureModel();
    fea_model -> LoadModel(modelfile + ".fea");
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    
    ((BaseLearner*)(this)) -> Init();
}

