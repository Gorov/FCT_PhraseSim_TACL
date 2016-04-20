//
//  MTLTrainer.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-24.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include "MTLTrainer.h"

void MTLTrainer::Init(char* embfile, char* clusfile, char* traindata_b, char* traindata_p, char* traindata_s)
{
    b_inst = new BinaryInstance();
    p_inst = new PPDBB2UInstance();
    s_inst = new SkipgramInstance();
    b_inst_num = 0;
    p_inst_num = 0;
    s_inst_num = 0;
    
    if (true) {
    fea_model = new FeatureModel();
    fea_model -> dim = emb_model -> layer1_size;
    fea_model -> InitClusDict(clusfile);
    fea_model -> lex_fea = true;
        if (strcmp(traindata_b, "") != 0) {
            b_inst_num = fea_model -> AddFeatDict(traindata_b, BINARY_INST);
        }
        if (strcmp(traindata_p, "") != 0) {
            p_inst_num = fea_model -> AddFeatDict(traindata_p, PPDB_INST);
        }
        if (strcmp(traindata_s, "") != 0) {
            s_inst_num = fea_model -> AddFeatDict(traindata_s, SKIPGRAM_INST);
        }
        fea_model -> InitFeatPara();
    }
    else {
        fea_model = new FeatureModel();
        string fea_file(embfile);
        fea_file = fea_file.substr(0, fea_file.size() - 3) + "fea";
        fea_model -> LoadModel(fea_file);
        fea_model -> InitClusDict(clusfile);
        fea_model -> lex_fea = true;
        if (strcmp(traindata_b, "") != 0) {
            b_inst_num = fea_model -> AddFeatDict(traindata_b, BINARY_INST);
        }
        if (strcmp(traindata_p, "") != 0) {
            p_inst_num = fea_model -> AddFeatDict(traindata_p, PPDB_INST);
        }
        if (strcmp(traindata_s, "") != 0) {
            s_inst_num = fea_model -> AddFeatDict(traindata_s, SKIPGRAM_INST);
        }
        fea_model -> InitAddtionalFeatPara();
    }
    
    num_fea = fea_model -> num_fea;
    fea_model -> num_inst = b_inst_num + p_inst_num + s_inst_num;
    b = 0.0;
}

void MTLTrainer::LoadMaintaskVocab(string trainfile, int type) {
    word2int::iterator iter;
    
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs, type)) {
        iter = maintask_vocab.find(inst->word1);
        if (iter == maintask_vocab.end()) {
            maintask_vocab[inst->word1] = 1;
        }
        iter = maintask_vocab.find(inst->word2);
        if (iter == maintask_vocab.end()) {
            maintask_vocab[inst->word2] = 1;
        }
        if (type == BINARY_INST) {
            iter = maintask_vocab.find(b_inst -> label);
            if (iter == maintask_vocab.end()) {
                maintask_vocab[b_inst->label] = 1;
            }
        }
    }
    ifs.close();
}

void MTLTrainer::EvalData(string trainfile, int type) {
    int total = 0;
    int right = 0;
    real loss = 0.0;
    long l2;
    
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs, type)) {
        ForwardProp(type);
        if (type == BINARY_INST || type == BINARY_INST_DS) {
            if (((BinaryInstance*)inst) -> score >= 0.5 && ((BinaryInstance*)inst) -> positive == 1) {
                right++;
            }
            else if (((BinaryInstance*)inst) -> score < 0.5 && ((BinaryInstance*)inst) -> positive == 0) {
                right++;
            }
        }
//        if (type == SKIPGRAM_INST_DS) {
//            for (int c = 0; c < s_inst -> num; c++) {
//                l2 = s_inst -> pos_ids[c] * layer1_size;
//                for (int a = 0; a < layer1_size; a++) loss += (emb_p[a] - emb_model->syn1neg[a]) * (emb_p[a] - emb_model->syn1neg[a]);
//            }
//        }
        if (type == SKIPGRAM_INST_DS) {
            int a, c;
            long l1;
            
            if (inst -> id1 == -1 && inst -> id2 == -1) {
                continue;
            }
            
            norm1 = 0.0;
            for (a = 0; a < layer1_size; a++) norm1 += emb_p[a] * emb_p[a];
            norm1 = sqrt(norm1);
            for (a = 0; a < layer1_size; a++) emb_p[a] /= norm1;
            for (c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
                if (((SkipgramInstance*)inst)->pos_ids[c] < 0) continue;
                l1 = ((SkipgramInstance*)inst)->pos_ids[c] * layer1_size;
                norm2 = 0.0;
                for (a = 0; a < layer1_size; a++) norm2 += emb_model->syn1neg[a + l1] * emb_model->syn1neg[a + l1];
                norm2 = sqrt(norm2);
                for (a = 0; a < layer1_size; a++) e2_norm[a] = (emb_model->syn1neg[a + l1]) / norm2;
                
                inner_norm = 0.0;
                for (a = 0; a < layer1_size; a++) inner_norm += emb_p[a] * e2_norm[a];
//                if (isnan(inner_norm)) {
//                    cout << "here" << endl;
//                }
                loss += inner_norm;
            }
        }
        total++;
    }
    if (type == BINARY_INST || type == BINARY_INST_DS) {
        cout << "Acc: " << (float)right / total << endl;
    }    
    if (type == SKIPGRAM_INST_DS) {
        cout << "Loss: " << loss / total << endl;
    }
    ifs.close();
}

void MTLTrainer::EvalMRR(string testfile, int threshold)
{
    long long size, b, c;
    
    int Total = 0;
    double score_total = 0.0;
    long long l1;
    real sim;
    real norm1, norm2;
    ifstream ifs(testfile.c_str());
    while (LoadInstance(ifs, PPDB_INST)) {
        Total++;
        ForwardProp(PPDB_INST);
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

void MTLTrainer::EvalLogLoss(string trainfile) {
    int total = 0;
    double loss = 0.0;
    next_random = 0;
    int count = 0;
    real tmploss = 0.0;
    int id, d;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs, SKIPGRAM_INST)) {
        count++;
        if (inst -> id1 < 0 && inst -> id2 < 0) continue;
        ForwardProp(SKIPGRAM_INST);
        for (int c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
            if (((SkipgramInstance*)inst)->pos_ids[c] < 0) continue;
            total ++;
            loss += log(((SkipgramInstance*)inst) -> pos_scores[c]);
        }
    }
    cout << "NCE Loss: " << loss / total << endl;
    ifs.close();
}

int MTLTrainer::LoadInstance(ifstream &ifs, int type) {
    char line_buf[1000];
    word2int::iterator iter;
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    if (type == BINARY_INST || type == BINARY_INST_DS) {
        inst = b_inst;
    }
    if (type == PPDB_INST) {
        inst = p_inst;
    }
    if (type == SKIPGRAM_INST || type == SKIPGRAM_INST_DS) {
        inst = s_inst;
    }
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
    
    if (type == BINARY_INST || type == BINARY_INST_DS) {
        ifs.getline(line_buf, 1000, '\n');
        iss.clear();
        iss.str(line_buf);
        iss >> b_inst -> label;
        ifs.getline(line_buf, 1000, '\n');
        iss.clear();
        iss.str(line_buf);
        iss >> b_inst -> positive;
    }
    if (type == PPDB_INST) {
        ifs.getline(line_buf, 1000, '\n');
        iss.clear();
        iss.str(line_buf);
        
        iss >> p_inst -> pos_label;
        iter = emb_model -> vocabdict.find(p_inst -> pos_label);
        if (iter == emb_model -> vocabdict.end()) {
            p_inst -> pos_id = -1;
        }
        else {
            p_inst -> pos_id = iter -> second;
        }
        p_inst -> pos_score = 0.0;
        
        for (int j = 0; j < 15; j++) {
            long id = SampleNegative();
            while (id == p_inst -> pos_id) {
                id = SampleNegative();
            }
            string word = vocab[id];
            p_inst -> neg_labels[j] = word;
            p_inst -> neg_ids[j] = id;
            p_inst -> neg_scores[j] = 0.0;
        }
    }
    if (type == SKIPGRAM_INST || type == SKIPGRAM_INST_DS) {
        ifs.getline(line_buf, 1000, '\n');
        iss.clear();
        iss.str(line_buf);
        iss >> s_inst -> num;
        for (int i = 0; i < s_inst -> num; i++) {
            iss >> s_inst -> pos_labels[i];
            iter = emb_model -> vocabdict.find(s_inst -> pos_labels[i]);
            if (iter == emb_model -> vocabdict.end()) {
                s_inst -> pos_ids[i] = -1;
            }
            else {
                s_inst -> pos_ids[i] = iter -> second;
            }
            s_inst -> pos_scores[i] = 0.0;
            
            for (int j = 0; j < 15; j++) {
                long id = SampleNegative();
                while (id == s_inst -> pos_ids[i]) {
                    id = SampleNegative();
                }
                string word = vocab[id];
                s_inst -> neg_labels[i * 15 + j] = word;
                s_inst -> neg_ids[i * 15 + j] = id;
                s_inst -> neg_scores[i * 15 + j] = 0.0;
            }
        }
    }
    
    return 1;
}

void MTLTrainer::ForwardOutputs(int type)
{
    int a, c, id;
    long long l1;
    double sum;
    word2int::iterator iter;
    
    if (type == BINARY_INST) {
        sum = 0.0;
        iter = emb_model -> vocabdict.find(((BinaryInstance*)inst)->label);
        if (iter == emb_model -> vocabdict.end()) {
            ((BinaryInstance*)inst) -> label_id = -1;
            ((BinaryInstance*)inst) -> score = 0.5;
        }
        else {
            ((BinaryInstance*)inst) -> label_id = iter -> second;
            l1 = iter->second * layer1_size;
            for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn0[a + l1];
            sum += b;
            ((BinaryInstance*)inst) -> score = Sigmoid(sum);
        }
    }
    if (type == BINARY_INST_DS) {
        sum = 0.0;
        iter = emb_model -> vocabdict.find(((BinaryInstance*)inst)->label);
        if (iter == emb_model -> vocabdict.end()) {
            ((BinaryInstance*)inst) -> label_id = -1;
            ((BinaryInstance*)inst) -> score = 0.5;
        }
        else {
            ((BinaryInstance*)inst) -> label_id = iter -> second;
            l1 = iter->second * layer1_size;
            for (a = 0; a < layer1_size; a++) sum += emb_p[a] * syn0[a + l1];
            sum += b;
            ((BinaryInstance*)inst) -> score = Sigmoid(sum);
        }
    }
    if  (type == PPDB_INST) {
        sum = 0.0;
        if (((PPDBB2UInstance*)inst)->pos_id < 0) {
            ((PPDBB2UInstance*)inst) -> pos_score = 0.0;
            return;
        }
        l1 = ((PPDBB2UInstance*)inst)->pos_id * layer1_size;
        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn0[a + l1];
        ((PPDBB2UInstance*)inst) -> pos_score = exp(sum);
        
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
        
        sum = ((PPDBB2UInstance*)inst) -> pos_score;
        for (int j = 0; j < 15; j++) {
            sum += ((PPDBB2UInstance*)inst) -> neg_scores[j];
        }
        ((PPDBB2UInstance*)inst) -> pos_score /= sum;
        for (int j = 0; j < 15; j++) {
            ((PPDBB2UInstance*)inst) -> neg_scores[j] /= sum;
        }
    }
//    if (type == PPDB_INST) {
//        sum = 0.0;
//        if (((PPDBB2UInstance*)inst)->pos_id < 0) {
//            ((PPDBB2UInstance*)inst) -> pos_score = 0.0;
//            return;
//        }
//        
//        norm1 = 0.0;
//        for (a = 0; a < layer1_size; a++) norm1 += emb_p[a] * emb_p[a];
//        norm1 = sqrt(norm1);
//        for (a = 0; a < layer1_size; a++) emb_p[a] /= norm1;
//        
//        l1 = ((PPDBB2UInstance*)inst)->pos_id * layer1_size;
//        norm2 = 0.0;
//        for (a = 0; a < layer1_size; a++) norm2 += emb_model->syn0[a + l1] * emb_model->syn0[a + l1];
//        norm2 = sqrt(norm2);
//        norms[0] = norm2;
//        for (a = 0; a < layer1_size; a++) e_norms[a] = emb_model->syn0[a + l1] / norm2;
//        
//        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * e_norms[a];
//        scores[0] = sum;
//        ((PPDBB2UInstance*)inst) -> pos_score = exp(sum);
//        
//        for (int j = 0; j < 15; j++) {
//            sum = 0.0;
//            if (((PPDBB2UInstance*)inst)->neg_ids[j] < 0) {
//                ((PPDBB2UInstance*)inst) -> neg_scores[j] = 0.0;
//                continue;
//            }
//            l1 = ((PPDBB2UInstance*)inst)->neg_ids[j]  * layer1_size;
//            norm2 = 0.0;
//            for (a = 0; a < layer1_size; a++) norm2 += emb_model->syn0[a + l1] * emb_model->syn0[a + l1];
//            norm2 = sqrt(norm2);
//            norms[j + 1] = norm2;
//            for (a = 0; a < layer1_size; a++) e_norms[a + (j + 1) * layer1_size] = emb_model->syn0[a + l1] / norm2;
//            
//            for (a = 0; a < layer1_size; a++) sum += emb_p[a] * e_norms[a + (j+1)*layer1_size];
//            scores[j+1] = sum;
//            ((PPDBB2UInstance*)inst) -> neg_scores[j] = exp(sum);
//        }
//        
//        sum = ((PPDBB2UInstance*)inst) -> pos_score;
//        for (int j = 0; j < 15; j++) {
//            sum += ((PPDBB2UInstance*)inst) -> neg_scores[j];
//        }
//        ((PPDBB2UInstance*)inst) -> pos_score /= sum;
//        for (int j = 0; j < 15; j++) {
//            ((PPDBB2UInstance*)inst) -> neg_scores[j] /= sum;
//        }
//    }
//    if (false){// (type == PPDB_INST) {
//        sum = 0.0;
//        if (((PPDBB2UInstance*)inst)->pos_id < 0) {
//            ((PPDBB2UInstance*)inst) -> pos_score = 0.0;
//            return;
//        }
//        l1 = ((PPDBB2UInstance*)inst)->pos_id * layer1_size;
//        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn0[a + l1];
//        ((PPDBB2UInstance*)inst) -> pos_score = 1.0 / (1 + exp(-sum));
//        
//        for (int j = 0; j < 15; j++) {
//            sum = 0.0;
//            if (((PPDBB2UInstance*)inst)->neg_ids[j] < 0) {
//                ((PPDBB2UInstance*)inst) -> neg_scores[j] = 0.0;
//                continue;
//            }
//            l1 = ((PPDBB2UInstance*)inst)->neg_ids[j]  * layer1_size;
//            for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn0[a + l1];
//            ((PPDBB2UInstance*)inst) -> neg_scores[j] = 1.0 / (1 + exp(-sum));
//        }
//    }
    if (type == SKIPGRAM_INST) {
        //word2int::iterator iter;
        for (c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
            if (((SkipgramInstance*)inst)->pos_ids[c] < 0) {
                ((SkipgramInstance*)inst) -> pos_scores[c] = 0.0;
                continue;
            }
            id = (int)((SkipgramInstance*)inst)->pos_ids[c];
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
    if (type == SKIPGRAM_INST_DS) {
    }
}

long MTLTrainer::BackPropPhrase(int type)
{
    int a, c, y;
    long long l1;
    long tmpid;
    for (a = 0; a < layer1_size; a++) part_emb_p[a] = 0.0;
    
    if (type == BINARY_INST) {
        tmpid = ((BinaryInstance*)inst) -> label_id;
        if (tmpid < 0) {
            return -1;
        }
        y = ((BinaryInstance*)inst)->positive;
        l1 = tmpid * layer1_size;
        
        for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((BinaryInstance*)inst)->score) * emb_model->syn0[a + l1];
        if (update_emb) {
            for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * (y - ((BinaryInstance*)inst)->score) * emb_p[a];
        }
        b += eta * (y - ((BinaryInstance*)inst)->score);
        return tmpid;
    }
    if (type == BINARY_INST_DS) {
        tmpid = ((BinaryInstance*)inst) -> label_id;
        if (tmpid < 0) {
            return -1;
        }
        y = ((BinaryInstance*)inst)->positive;
        l1 = tmpid * layer1_size;
        
        for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((BinaryInstance*)inst)->score) * syn0[a + l1];
        //if (update_emb) {
        //    for (a = 0; a < layer1_size; a++) syn0[a + l1] += eta * (y - ((BinaryInstance*)inst)->score) * emb_p[a];
        //}
        b += eta * (y - ((BinaryInstance*)inst)->score);
        return tmpid;
    }
    if (type == PPDB_INST) {
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
//    if (type == PPDB_INST) {
//        tmpid = ((PPDBB2UInstance*)inst) -> pos_id;
//        if (tmpid < 0) {
//            return -1;
//        }
//        y = 1;
//        l1 = tmpid * layer1_size;
//        for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((PPDBB2UInstance*)inst)->pos_score) * (e_norms[a] / norm1 - scores[0] * emb_p[a] / norm1);
//        if (update_emb) {
//            for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * (y - ((PPDBB2UInstance*)inst)->pos_score) * (emb_p[a] / norms[0] - scores[0] * e_norms[a] / norms[0]);
//        }
//        
//        for (int j = 0; j < 15; j++) {
//            tmpid = ((PPDBB2UInstance*)inst) -> neg_ids[j];
//            if (tmpid < 0) {
//                continue;
//            }
//            y = 0;
//            l1 = tmpid * layer1_size;
//            for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((PPDBB2UInstance*)inst)->neg_scores[j]) * (e_norms[a + (j+1)*layer1_size] / norm1 - scores[j+1] * emb_p[a] / norm1);
//            if (update_emb) {
//                for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * (y - ((PPDBB2UInstance*)inst)->neg_scores[j]) * (emb_p[a] / norms[j+1] - scores[j+1] * e_norms[a + (j+1) * layer1_size] / norms[j+1]);
//            }
//        }
//        return 0;
//    }
    if (type == SKIPGRAM_INST) {
        for (c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
            tmpid = ((SkipgramInstance*)inst) -> pos_ids[c];
            if (tmpid < 0) {
                continue;
            }
                y = 1;
                l1 = tmpid * layer1_size;
                for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((SkipgramInstance*)inst)->pos_scores[c]) * emb_model->syn1neg[a + l1];
                if (update_emb) {
                    for (a = 0; a < layer1_size; a++) emb_model->syn1neg[a + l1] += eta * C * (y - ((SkipgramInstance*)inst)->pos_scores[c]) * emb_p[a];
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
                        for (a = 0; a < layer1_size; a++) emb_model->syn1neg[a + l1] += eta * C * (y - ((SkipgramInstance*)inst)->neg_scores[c * 15 + j]) * emb_p[a];
                    }
                }
        }
    }
    if (false) {//if (type == SKIPGRAM_INST_DS) {
        for (c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
            tmpid = ((SkipgramInstance*)inst) -> pos_ids[c];
            if (tmpid < 0) {
                continue;
            }
            l1 = tmpid * layer1_size;
            for (a = 0; a < layer1_size; a++) part_emb_p[a] += emb_model->syn1neg[a + l1] - emb_p[a];
        }
    }
    if (type == SKIPGRAM_INST_DS) {
        norm1 = 0.0;
        for (a = 0; a < layer1_size; a++) norm1 += emb_p[a] * emb_p[a];
        norm1 = sqrt(norm1);
        for (a = 0; a < layer1_size; a++) emb_p[a] /= norm1;
        for (c = 0; c < ((SkipgramInstance*)inst) -> num; c++) {
            if (((SkipgramInstance*)inst)->pos_ids[c] < 0) continue;
            l1 = ((SkipgramInstance*)inst)->pos_ids[c] * layer1_size;
            norm2 = 0.0;
            for (a = 0; a < layer1_size; a++) norm2 += emb_model->syn1neg[a + l1] * emb_model->syn1neg[a + l1];
            norm2 = sqrt(norm2);
            for (a = 0; a < layer1_size; a++) e2_norm[a] = emb_model->syn1neg[a + l1] / norm2;

            inner_norm = 0.0;
            for (a = 0; a < layer1_size; a++) inner_norm += emb_p[a] * e2_norm[a];
            for (a = 0; a < layer1_size; a++) part_emb_p[a] += (e2_norm[a] / norm1) - (inner_norm * emb_p[a] / norm1);
        }
    }
}

double MTLTrainer::Sigmoid(double score) {
    return 1.0 / (1 + exp(-score));
}

void MTLTrainer::TrainBigData(char* traindata_b, char* traindata_s, char* dev_file_b) {
    int count = 0;
    int batchsize = 10;
    eta = eta0;
    int max_inst = max(b_inst_num, s_inst_num);
    if (max_inst == 0) return;
    int b_freq, s_freq;
    if (b_inst_num == 0) {
        b_freq = 0;
        s_freq = batchsize;
    }
    else if(s_inst_num == 0) {
        s_freq = 0;
        b_freq = batchsize;
    }
    else if (max_inst == b_inst_num) {
        s_freq = batchsize;
        b_freq = max_inst * batchsize / s_inst_num;
    }
    else {
        b_freq = batchsize;
        s_freq = max_inst * batchsize / b_inst_num;
    }
    
    for (int i = 0; i < iter; i++) {
        cout << "Iter: " << i << endl;
        ifstream ifs_b, ifs_s;
        if (strcmp(traindata_b, "") != 0 ) {
            ifs_b.open(traindata_b);
        }
        if (strcmp(traindata_s, "") != 0 ) {
            ifs_s.open(traindata_s);
        }
        while (true) {
            int j, k;
            for (j = 0; j < b_freq; j++) {
                if(!LoadInstance(ifs_b, BINARY_INST_DS)) break;
                inst = b_inst;
                ForwardProp(BINARY_INST_DS);
                BackProp(BINARY_INST_DS);
                count++;
                if (count % 100000 == 0) {
                    cout << count << endl;
                    EvalData(dev_file_b, BINARY_INST_DS);
                    eta = eta0 * (1 - count / (double)(fea_model -> num_inst * iter + 1));
                    //eta = eta0;
                    cout << "Learning rate:" << eta << endl;
                }
            }
            for (k = 0; k < s_freq; k++) {
                if(!LoadInstance(ifs_s, SKIPGRAM_INST)) break;
                inst = s_inst;
                ForwardProp(SKIPGRAM_INST);
                BackProp(SKIPGRAM_INST);
                count++;
                if (count % 100000 == 0) {
                    cout << count << endl;
                    EvalData(dev_file_b, BINARY_INST_DS);
                    eta = eta0 * (1 - count / (double)(fea_model -> num_inst * iter + 1));
                    //eta = eta0;
                    cout << "Learning rate:" << eta << endl;
                }
            }
            int finish = 0;
            if (b_freq == 0) finish ++;
            else if (j != b_freq) finish++;
            if (s_freq == 0) finish ++;
            else if (k != s_freq) finish++;
            if (finish == 2) {
                break;
            }
            //eta = eta0 * (1 - count / (double)(iter * (b_inst_num + p_inst_num) + 1));
            //eta = eta0;
            //if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        }
        ifs_b.close();
        ifs_s.close();
    }
    printf("b1: %lf\n", fea_model -> b1s[0]);
    printf("b2: %lf\n", fea_model -> b2s[0]);
}

void MTLTrainer::TrainBigDataNew(char* traindata_b, char* traindata_s, char* dev_file_b, char* test_file_b, int type) {
    int count = 0;
    int batchsize = 10;
    eta = eta0;
    int tmp_inst_num = b_inst_num;
    if (type == PPDB_INST) {
        tmp_inst_num = p_inst_num;
    }
    int max_inst = max(tmp_inst_num, s_inst_num);
    if (max_inst == 0) return;
    int b_freq, s_freq;
    if (tmp_inst_num == 0) {
        b_freq = 0;
        s_freq = batchsize;
    }
    else if(s_inst_num == 0) {
        s_freq = 0;
        b_freq = batchsize;
    }
    else if (max_inst == tmp_inst_num) {
        s_freq = batchsize;
        b_freq = max_inst * batchsize / s_inst_num;
    }
    else {
        b_freq = batchsize;
        s_freq = max_inst * batchsize / tmp_inst_num;
    }
    
    for (int i = 0; i < iter; i++) {
        cout << "Iter: " << i << endl;
        ifstream ifs_b, ifs_s;
        if (strcmp(traindata_b, "") != 0 ) {
            ifs_b.open(traindata_b);
        }
        if (strcmp(traindata_s, "") != 0 ) {
            ifs_s.open(traindata_s);
        }
        while (true) {
            int j, k;
            for (j = 0; j < b_freq; j++) {
                if(!LoadInstance(ifs_b, type)) break;
                if (inst -> id1 == -1 && inst -> id2 == -1) {
                    continue;
                }
                if (type == BINARY_INST || type == BINARY_INST_DS) inst = b_inst;
                else inst = p_inst;
                ForwardProp(type);
                BackProp(type);
                count++;
                if (count % 100000 == 0) {
                    cout << count << endl;
                    if (type == BINARY_INST || type == BINARY_INST_DS) {
                        EvalData(dev_file_b, BINARY_INST_DS);
                    }
                    else EvalMRR(dev_file_b, 1000);
                    
                    eta = eta0 * (1 - count / (double)(fea_model -> num_inst * iter + 1));
                    //eta = eta0;
                    cout << "Learning rate:" << eta << endl;
                }
            }
            for (k = 0; k < s_freq; k++) {
                if(!LoadInstance(ifs_s, SKIPGRAM_INST)) break;
                inst = s_inst;
                ForwardProp(SKIPGRAM_INST);
                BackProp(SKIPGRAM_INST);
                count++;
                if (count % 100000 == 0) {
                    cout << count << endl;
                    if (type == BINARY_INST || type == BINARY_INST_DS) {
                        EvalData(dev_file_b, BINARY_INST_DS);
                    }
                    else {
                        EvalMRR(dev_file_b, 1000);
                        EvalMRR(test_file_b, 1000);
                        EvalMRR(dev_file_b, 10000);
                        EvalMRR(test_file_b, 10000);
                        EvalMRR(dev_file_b, 100000);
                        EvalMRR(test_file_b, 100000);
                    }
                    eta = eta0 * (1 - count / (double)(fea_model -> num_inst * iter + 1));
                    //eta = eta0;
                    cout << "Learning rate:" << eta << endl;
                }
            }
            int finish = 0;
            if (b_freq == 0) finish ++;
            else if (j != b_freq) finish++;
            if (s_freq == 0) finish ++;
            else if (k != s_freq) finish++;
            if (finish == 2) {
                break;
            }
            //eta = eta0 * (1 - count / (double)(iter * (b_inst_num + p_inst_num) + 1));
            //eta = eta0;
            //if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        }
        ifs_b.close();
        ifs_s.close();
    }
    cout << count << endl;
    printf("b1: %lf\n", fea_model -> b1s[0]);
    printf("b2: %lf\n", fea_model -> b2s[0]);
}

void MTLTrainer::TrainDataNew(char* traindata_b, char* traindata_s) {
    int count = 0;
    int batchsize = 10;
    int max_inst = max(b_inst_num, s_inst_num);
    if (max_inst == 0) return;
    int b_freq, s_freq;
    if (b_inst_num == 0) {
        b_freq = 0;
        s_freq = batchsize;
    }
    else if(s_inst_num == 0) {
        s_freq = 0;
        b_freq = batchsize;
    }
    else if (max_inst == b_inst_num) {
        s_freq = batchsize;
        b_freq = max_inst * batchsize / s_inst_num;
    }
    else {
        b_freq = batchsize;
        s_freq = max_inst * batchsize / b_inst_num;
    }
    
    for (int i = 0; i < iter; i++) {
        cout << "Iter: " << i << endl;
        ifstream ifs_b, ifs_s;
        if (strcmp(traindata_b, "") != 0 ) {
            ifs_b.open(traindata_b);
        }
        if (strcmp(traindata_s, "") != 0 ) {
            ifs_s.open(traindata_s);
        }
        while (true) {
            int j, k;
            for (j = 0; j < b_freq; j++) {
                if(!LoadInstance(ifs_b, BINARY_INST)) break;
                inst = b_inst;
                ForwardProp(BINARY_INST);
                BackProp(BINARY_INST);
                count++;
            }
            for (k = 0; k < s_freq; k++) {
                if(!LoadInstance(ifs_s, SKIPGRAM_INST)) break;
                inst = s_inst;
                ForwardProp(SKIPGRAM_INST);
                BackProp(SKIPGRAM_INST);
                count++;
            }
            int finish = 0;
            if (b_freq == 0) finish ++;
            else if (j != b_freq) finish++;
            if (s_freq == 0) finish ++;
            else if (k != s_freq) finish++;
            if (finish == 2) {
                break;
            }
            //eta = eta0 * (1 - count / (double)(iter * (b_inst_num + p_inst_num) + 1));
            eta = eta0;
            if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        }
        ifs_b.close();
        ifs_s.close();
    }
    cout << count << endl;
    printf("b1: %lf\n", fea_model -> b1s[0]);
    printf("b2: %lf\n", fea_model -> b2s[0]);
}

void MTLTrainer::TrainDataNew(char* traindata_s) {
    int count = 0;
    int s_freq = s_inst_num;
    
    for (int i = 0; i < iter; i++) {
        cout << "Iter: " << i << endl;
        ifstream ifs_s;
        if (strcmp(traindata_s, "") != 0 ) {
            ifs_s.open(traindata_s);
        }

            for (int k = 0; k < s_freq; k++) {
                if(!LoadInstance(ifs_s, SKIPGRAM_INST_DS)) break;
                inst = s_inst;
                ForwardProp(SKIPGRAM_INST_DS);
                BackProp(SKIPGRAM_INST_DS);
                count++;
            }
            //eta = eta0 * (1 - count / (double)(iter * (b_inst_num + p_inst_num) + 1));
            eta = eta0;
            if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs_s.close();
    }
    cout << count << endl;
    printf("b1: %lf\n", fea_model -> b1s[0]);
    printf("b2: %lf\n", fea_model -> b2s[0]);
}

void MTLTrainer::TrainData(char* traindata_b, char* traindata_p, char* traindata_s) {
    int count = 0;
    int batchsize = 10;
    int max_inst = max(p_inst_num, s_inst_num);
    if (max_inst == 0) return;
    int b_freq, s_freq, p_freq;
    if (s_inst_num == 0) {
        s_freq = 0;
        p_freq = batchsize;
    }
    else if(p_inst_num == 0) {
        p_freq = 0;
        s_freq = batchsize;
    }
    else if (max_inst == s_inst_num) {
        p_freq = batchsize;
        s_freq = max_inst * batchsize / p_inst_num;
    }
    else {
        s_freq = batchsize;
        p_freq = max_inst * batchsize / s_inst_num;
    }
    
    for (int i = 0; i < iter; i++) {
        ifstream ifs_b, ifs_p, ifs_s;
        if (strcmp(traindata_s, "") != 0 ) {
            ifs_s.open(traindata_s);
        }
        if (strcmp(traindata_p, "") != 0 ) {
            ifs_p.open(traindata_p);
        }
        while (true) {
            int j, k, l;
            for (j = 0; j < b_freq; j++) {
//                if(!LoadInstance(ifs_b, BINARY_INST)) break;
//                inst = b_inst;
//                ForwardProp(BINARY_INST);
//                BackProp(BINARY_INST);
                if(!LoadInstance(ifs_b, BINARY_INST_DS)) break;
                inst = b_inst;
                ForwardProp(BINARY_INST_DS);
                BackProp(BINARY_INST_DS);
                count++;
            }
            for (k = 0; k < p_freq; k++) {
                if(!LoadInstance(ifs_p, PPDB_INST)) break;
                inst = p_inst;
                ForwardProp(PPDB_INST);
                BackProp(PPDB_INST);
                count++;
            }
            for (l = 0; l < s_freq; l++) {
                if(!LoadInstance(ifs_s, SKIPGRAM_INST)) break;
                inst = s_inst;
                ForwardProp(SKIPGRAM_INST);
                BackProp(SKIPGRAM_INST);
                count++;
            }
            int finish = 0;
            if (s_freq == 0) finish ++;
            else if (l != s_freq) finish++;
            if (p_freq == 0) finish ++;
            else if (k != p_freq) finish++;
            if (finish == 2) {
                break;
            }
            //eta = eta0 * (1 - count / (double)(iter * (b_inst_num + p_inst_num) + 1));
            eta = eta0;
            if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        }
        ifs_b.close();
        ifs_p.close();
        ifs_s.close();
    }
    cout << count << endl;
    printf("b1: %lf\n", fea_model -> b1s[0]);
    printf("b2: %lf\n", fea_model -> b2s[0]);
}

//void MTLTrainer::TrainData(char* traindata_b, char* traindata_p, char* traindata_s) {
//    int count = 0;
//    for (int i = 0; i < iter; i++) {
//        ifstream ifs;
//        if (strcmp(traindata_b, "") != 0 ) {
//            ifs.open(traindata_b);
//            while (LoadInstance(ifs, BINARY_INST)) {
//                inst = b_inst;
//                ForwardProp(BINARY_INST);
//                BackProp(BINARY_INST);
//                count++;
//            }
//            //eta = eta0 * (1 - i / (double)(iter + 1));
//            eta = eta0;
//            if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
//            ifs.close();
//        }
//        
//        if (strcmp(traindata_p, "") != 0 ) {
//            ifs.open(traindata_p);
//            while (LoadInstance(ifs, PPDB_INST)) {
//                inst = p_inst;
//                ForwardProp(PPDB_INST);
//                BackProp(PPDB_INST);
//                count++;
//            }
//            ifs.close();
//        }
//    }
//    printf("b1: %lf\n", fea_model -> b1s[0]);
//    printf("b2: %lf\n", fea_model -> b2s[0]);
//}

void MTLTrainer::ForwardProp(int type)
{
    int a, c;
    long long l1;
    real sum;
    
    ExtractFeature();
    ComputeLambdas();
    
    if (type == SKIPGRAM_INST_DS) {
        if (inst -> id1 == -1 && inst -> id2 == -1) {
            return;
        }
    }
    
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
    ForwardOutputs(type);
}

void MTLTrainer::BackProp(int type)
{
    int a;
    long long l1;
    word2int::iterator iter;
    if (type == SKIPGRAM_INST_DS) {
        if (inst -> id1 == -1 && inst -> id2 == -1) return;
    }
    for (a = 0; a < layer1_size; a++) part_emb_p[a] = 0.0;
    long tmpid = BackPropPhrase(type);
    if (tmpid < 0) return;
    
    for (a = 0; a < layer1_size; a++) part_lambda1[a] = 0.0;
    for (a = 0; a < layer1_size; a++) part_lambda2[a] = 0.0;
    if (inst -> id1 >= 0) {
        l1 = inst -> id1 * layer1_size;
        for (a = 0; a < layer1_size; a++) {
            part_lambda1[a] = part_emb_p[a] * emb_model->syn0[a + l1];
        }
        if (update_emb) 
        {
            if (type != SKIPGRAM_INST && type != SKIPGRAM_INST_DS) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * part_emb_p[a] * lambda1[a];
            else if (type == SKIPGRAM_INST_DS) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * part_emb_p[a] * lambda1[a];
            else {
                //for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda1[a];
                if (!mask) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda1[a];
                else {
                    iter = maintask_vocab.find(inst -> word1);
                    //if (iter == maintask_vocab.end()) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda1[a];
                    for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda1[a];
                    //if (iter != maintask_vocab.end()) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda1[a];
                }
            }
        }
    }
    if (inst -> id2 >= 0) {
        l1 = inst -> id2 * layer1_size;
        for (a = 0; a < layer1_size; a++) {
            part_lambda2[a] = part_emb_p[a] * emb_model->syn0[a + l1];
        }
//        if (update_emb) for (a = 0; a < layer1_size; a++) {
//            if (type != SKIPGRAM_INST) emb_model->syn0[a + l1] += eta * part_emb_p[a] * lambda2[a];
//            else emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda2[a];
//        }
        if (update_emb) 
        {
            if (type != SKIPGRAM_INST && type != SKIPGRAM_INST_DS) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * part_emb_p[a] * lambda2[a];
            else if (type == SKIPGRAM_INST_DS) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * part_emb_p[a] * lambda1[a];
            else {
                //for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda2[a];
                if (!mask) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda2[a];
                else {
                    iter = maintask_vocab.find(inst -> word2);
                    //if (iter == maintask_vocab.end()) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda2[a];
                    for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda2[a];
                    //if (iter != maintask_vocab.end()) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta * C * part_emb_p[a] * lambda2[a];
                }
            }
        }
    }
    if (type != SKIPGRAM_INST && type != SKIPGRAM_INST_DS) {
        BackPropFeatures(1.0);
    }
    else if (type == SKIPGRAM_INST_DS) BackPropFeatures(1.0);
    else {
        if (!mask) BackPropFeatures(C);
    }
    
}

void MTLTrainer::BackPropFeatures(real C) {
    int a, c;
    for (c = 0; c < layer1_size; c++) {
        for (a = 0; a < num_feat1; a++) {
            fea_model -> param[c * num_fea + feat_vec1[a]] += eta * C * part_lambda1[c];
        }
        fea_model -> b1s[c] += eta * C * part_lambda1[c];
    }
    for (c = 0; c < layer1_size; c++) {
        for (a = 0; a < num_feat2; a++) {
            fea_model -> param[c * num_fea + feat_vec2[a]] += eta * C * part_lambda2[c];
        }
        fea_model -> b2s[c] += eta * C * part_lambda2[c];
    }
}

void MTLTrainer::InitFreqTable(char* filename) {
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
        }
        ifs.getline(line_buf, 1000, '\n');    
    }
    ifs.close();
}

void MTLTrainer::InitUnigramTable() {
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

long MTLTrainer::SampleNegative() {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    long ret = table[(next_random >> 16) % table_size];
    if (ret == 0) ret = next_random % (vocab_size - 1) + 1;
    //cout << next_random << " " << ret << " ";
    return ret;
}

void MTLTrainer::InitVocab(char* vocabfile) {
    next_random = 0;
    InitFreqTable(vocabfile);
    InitUnigramTable();
    
    vocab.resize(vocab_size);
    word2int::iterator iter;
    for (iter = emb_model -> vocabdict.begin(); iter != emb_model -> vocabdict.end(); iter++) {
        vocab[iter -> second] = iter -> first;
    }
}

