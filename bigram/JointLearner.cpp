//
//  JointLearner.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-17.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "JointLearner.h"

/*
void JointLearner::Init() {
    inst = new SkipgramInstance();
}

void JointLearner::Init(char* embfile, char* trainfile)
{
    inst = new SkipgramInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, 2);
    num_fea = fea_model -> num_fea;
}

void JointLearner::Init(char* embfile, char* clusfile, char* trainfile)
{
    inst = new SkipgramInstance();
    fea_model = new FeatureModel((int)layer1_size, trainfile, clusfile, 2);
    num_fea = fea_model -> num_fea;
    //init vocab and freq table
}

void JointLearner::InitWithOption(char* embfile, char* clusfile, char* trainfile)
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
}*/

void JointLearner::InitVocab() {
    next_random = 0;
    freqtable = (int*)malloc(sizeof(int) * emb_model -> vocabdict.size());
    for (int i = 0; i < emb_model -> vocabdict.size(); i++) {
        freqtable[i] = emb_model -> freqlist[i];
    }
    
    InitUnigramTable();
    
    vocab.resize(vocab_size);
    word2int::iterator iter;
    for (iter = emb_model -> vocabdict.begin(); iter != emb_model -> vocabdict.end(); iter++) {
        vocab[iter -> second] = iter -> first;
    }
}

void JointLearner::ForwardOutputs()
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

long JointLearner::BackPropPhrase() {
    int a, c, d,y;
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
        if (negative){
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

void * JointLearner::TrainWordEmbThread(long thread_id) {
    long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)thread_id;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(this -> layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(train_lm_file.c_str(), "rb");
    fseek(fi, 0, SEEK_END);
    long long file_size = ftell(fi);
    fseek(fi, file_size / (long long)num_threads * (long long)thread_id, SEEK_SET);
        
    while (1) {
        if (word_count - last_word_count > 10000) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            now=clock();
            printf("%cAlpha: %f  Progress: %.2f%%", 13, lm_alpha,
                   word_count_actual / (real)(train_words + 1) * 100);
            fflush(stdout);
            lm_alpha = alpha0 * (1 - word_count_actual / (real)(train_words + 1));
            if (lm_alpha < alpha0 * 0.0001) lm_alpha = alpha0 * 0.0001;
        }
        if (sentence_length == 0) {
            while (1) {
                word = ReadWordIndex(fi);
                if (feof(fi)) break;
                if (word == -1) continue;
                word_count++;
                if (word == 0) break;
                // The subsampling randomly discards frequent words while keeping the ranking same
                if (sample > 0) {
                    real ran = (sqrt(freqtable[word] / (sample * train_words)) + 1) * (sample * train_words) / freqtable[word];
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi)) break;
        if (word_count > train_words / num_threads) break;
        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        //else {  //train skip-gram
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                c = sentence_position - window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                l1 = last_word * layer1_size;
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                // HIERARCHICAL SOFTMAX
                if (hs) for (d = 0; d < wordinfo_table[word].codelen; d++) {
                    f = 0;
                    l2 = wordinfo_table[word].point[d] * layer1_size;
                    // Propagate hidden -> output
                    for (c = 0; c < layer1_size; c++) f += emb_model -> syn0[c + l1] * emb_model -> syn1[c + l2];
                    if (f <= -max_exp) continue;
                    else if (f >= max_exp) continue;
                    else f = expTable[(int)((f + max_exp) * (exp_table_size / max_exp / 2))];
                    // 'g' is the gradient multiplied by the learning rate
                    g = (1 - wordinfo_table[word].code[d] - f) * lm_alpha;
                    // Propagate errors output -> hidden
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * emb_model ->  syn1[c + l2];
                    // Learn weights hidden -> output
                    for (c = 0; c < layer1_size; c++) emb_model ->  syn1[c + l2] += g * emb_model ->  syn0[c + l1];
                }
                // NEGATIVE SAMPLING
                if (negative){
                for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = table[(next_random >> 16) % table_size];
                        if (target == 0) target = next_random % (vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    l2 = target * layer1_size;
                    f = 0;
                    for (c = 0; c < layer1_size; c++) { f += (emb_model -> syn0[c + l1]) * (emb_model -> syn1neg[c + l2]);}
                    if (f > max_exp) g = (label - 1) * lm_alpha;
                    else if (f < -max_exp) g = (label - 0) * lm_alpha;
                    else g = (label - expTable[(int)((f + max_exp) * (exp_table_size / max_exp / 2))]) * lm_alpha;
                    //else g = (label - (1.0 / (1 + exp(-f))) ) * lm_alpha;
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * (emb_model -> syn1neg[c + l2]);
                    for (c = 0; c < layer1_size; c++) (emb_model -> syn1neg[c + l2]) += g * (emb_model -> syn0[c + l1]);
                }
                }
                // Learn weights input -> hidden
                for (c = 0; c < layer1_size; c++) (emb_model -> syn0[c + l1]) += neu1e[c];
            }
        //}
        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void JointLearner::ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}

int JointLearner::ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    word2int::iterator iter = emb_model->vocabdict.find(word);
    if (iter == emb_model -> vocabdict.end()) return -1;
    else return iter-> second;
}

void JointLearner::InitTrainInfoLM(const char* filename)
{
    FILE* filein = fopen(filename, "r");
    train_words = 0;
    int word;
    while (1) {
        word = ReadWordIndex(filein);
        if (feof(filein)) break;
        if (word == -1) continue;
        train_words++;
    }
    cout << train_words << endl;
    word_count_actual = 0;
}

void JointLearner::InitTrainInfoLM()
{
    train_words = 0;
    for (int i = 0; i < emb_model -> freqlist.size(); i++) {
        train_words += emb_model -> freqlist[i];
    }
    cout << train_words << endl;
    word_count_actual = 0;
}

void JointLearner::InitExpTable() {
    int i;
    expTable = (real *)malloc((exp_table_size + 1) * sizeof(real));
    for (i = 0; i < exp_table_size; i++) {
        expTable[i] = exp((i / (real)exp_table_size * 2 - 1) * max_exp); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
}

void JointLearner::JointTrainBigData(string trainfile, string trainfile_lm, string trainsubfile, string devfile) {
    long a;
    InitExpTable();
    if(init_vocab) InitTrainInfoLM();
    else InitTrainInfoLM(train_lm_file.c_str());
    
    pthread_t *pt = (pthread_t *)malloc((num_threads) * sizeof(pthread_t));
    cout << "Starting training using file " << trainfile_lm << endl;
    
    for (a = 0; a < num_threads; a++) {
        ThreadPara* para = new ThreadPara(this, a);
        pthread_create(&pt[a], NULL, threadFunction, (void *)para);
    }
    if (false) {
        ThreadPara2* para2 = new ThreadPara2(this, trainfile);
        pthread_create(&pt[a], NULL, threadFunction2, (void *)para2);
        for (a = 0; a < num_threads + 1; a++) pthread_join(pt[a], NULL);
    }
    else for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    
    //cout << fea_model -> b1s[0] << endl;
    //cout << fea_model -> b2s[0] << endl;
}

void JointLearner::LoadModel(string modelfile) {
    emb_model = new EmbeddingModel((char*)(modelfile + ".emb").c_str(), LM);
    fea_model = new FeatureModel();
    fea_model -> LoadModel(modelfile + ".fea");
    
    layer1_size = emb_model -> layer1_size;
    num_fea = fea_model -> num_fea;
    
    ((BaseLearner*)(this)) -> Init();
}
