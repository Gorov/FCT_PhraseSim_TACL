//
//  PPDBM2M.cpp
//  Preposition_Classification
//
//  Created by gflfof gflfof on 15-1-14.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "PPDBM2M.h"

void PPDBM2M::Init(char* embfile, char* trainfile)
{
    fea_params.head = false;
    fea_params.position = false;
    fea_params.postag = false;
    fea_params.clus = false;
    fea_params.sum = true;
    
    fea_params.low_rank = true;
    
    emb_model = new EmbeddingModel(embfile);
    layer1_size = (int)emb_model -> layer1_size;
    emb_model_list.push_back(emb_model);
    
    BuildModelsFromData(trainfile);
    //    InitFeatureModel((int)layer1_size, trainfile, type);
    num_models = (int)deep_fct_list.size();
    num_labels = 16;
    label_scores = (real*) malloc(sizeof(real) * 16);
    label_embeddings = (real*) malloc(sizeof(real) * num_labels * layer1_size); 
    b_inst -> scores.resize(16);
    
    alpha = 1.0;
    lambda = 0.0;
}

void PPDBM2M::InitSubmodels() {
    for (int i = 0; i < deep_fct_list.size(); i++) {
        deep_fct_list[i] -> num_labels = 16;
        deep_fct_list[i] -> InitModel();
        deep_fct_list[i] -> update_emb = update_emb;
        deep_fct_list[i] -> adagrad = adagrad;
    }
}

void PPDBM2M::InitVocab(char* vocabfile) {
    next_random = 0;
    InitFreqTable(vocabfile);
    InitUnigramTable();
}

void PPDBM2M::BuildModelsFromData(char* trainfile) {
    layer1_size = (int)emb_model_list[0] -> layer1_size;
    
    ifstream ifs(trainfile);
    num_inst = 0;
    while (LoadInstanceInit(ifs)) {
        num_inst++;
    }
    
    ifs.close();
}

int PPDBM2M::LoadInstanceInit(ifstream &ifs) {
    char line_buf[5000];
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    
    {
        istringstream iss(line_buf);
        iss >> inst -> length;
        iss >> inst -> head;
        iss.clear();
        ifs.getline(line_buf, 5000, '\n');
        iss.str(line_buf);
        for (int i = 0; i < inst -> length; i++) {
            iss >> inst -> words[i];
        }
        iss.clear();
        ifs.getline(line_buf, 5000, '\n');
        iss.str(line_buf);
        for (int i = 0; i < inst -> length; i++) {
            iss >> inst -> tags[i];
        }
        iss.clear();
//        ifs.getline(line_buf, 5000, '\n');
//        iss.str(line_buf);
//        for (int i = 0; i < inst -> length; i++) {
//            iss >> inst -> clusters[i];
//        }
    }
    {
        int count = 0;
        int model_id = -1;
        string token, tag, slot_key;
        
        while (count < inst -> length) {
            slot_key = "PPDB_FCT";
            model_id = AddDeepFctModel(slot_key);
            
            if (fea_params.sum) {
                ostringstream oss;
                oss << "BIAS";
                slot_key = oss.str();
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            }
            
            if (fea_params.head)
            if (inst -> head == count) {
                ostringstream oss;
                oss << "HEAD";
                slot_key = oss.str();
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            }
            
            if (fea_params.postag){// && (count == 3 || count == 6)) {
                ostringstream oss;
                oss << "POS_i_" << inst -> tags[count];
                slot_key = oss.str();
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                
                oss.str("");
                if (count != 0) oss << "POS_i-1_" << inst -> tags[count - 1];
                else oss << "POS_i-1_BEGIN";
                slot_key = oss.str();
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                
                oss.str("");
                if (count != inst -> length - 1) oss << "POS_i+1_" << inst -> tags[count + 1];
                else oss << "POS_i+1_END";
                slot_key = oss.str();
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                
                oss.str("");
                oss << "POS_head_" << inst -> tags[inst -> head];
                slot_key = oss.str();
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                
                oss.str("");
                oss << "POS_head_" << inst -> tags[inst -> head] << "&POS_i_" << inst -> tags[count];
                slot_key = oss.str();
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                
                oss.str("");
                if (count != 0) oss << "POS_i-1_" << inst -> tags[count - 1] << "&POS_i_" << inst -> tags[count];
                else oss << "POS_i-1_BEGIN&POS_i_" << inst -> tags[count];
                slot_key = oss.str();
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                
                oss.str("");
                if (count != inst -> length - 1) oss << "POS_i_" << inst -> tags[count] << "&POS_i+1_" << inst -> tags[count + 1];
                else oss << "POS_i_" << inst -> tags[count] << "&POS_i+1_END";
                slot_key = oss.str();
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);            
            }
            
            count++;
        }
        
        if (inst -> length > max_len) {max_len = inst -> length;}
    }
    return 1;
}

int PPDBM2M::LoadInstance(ifstream &ifs) {
    word2int::iterator iter2;
    char line_buf[5000];
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    
    for (int i = 0; i < deep_fct_list.size(); i++) {
        deep_fct_list[i] -> inst -> Clear();
    }
    
    {
        istringstream iss(line_buf);
        iss >> inst -> length;
        iss >> inst -> head;
        iss.clear();
        ifs.getline(line_buf, 5000, '\n');
        iss.str(line_buf);
        for (int i = 0; i < inst -> length; i++) {
            iss >> inst -> words[i];
        }
        iss.clear();
        ifs.getline(line_buf, 5000, '\n');
        iss.str(line_buf);
        for (int i = 0; i < inst -> length; i++) {
            iss >> inst -> tags[i];
        }
        iss.clear();
//        ifs.getline(line_buf, 5000, '\n');
//        iss.str(line_buf);
//        for (int i = 0; i < inst -> length; i++) {
//            iss >> inst -> clusters[i];
//        }
    }
    ExtractFeatures();
    if (false)
    {
        int count = 0;
        int model_id = -1;
        int id;
        string token, tag, slot_key;
        FctDeepModel* p_model = NULL;
        RealFctPathInstance* p_inst = NULL;
        
        if (fea_params.low_rank) {
            slot_key = "PPDB_FCT";
            model_id = SearchDeepFctSlot(slot_key);
            p_model = deep_fct_list[model_id];
            p_inst = p_model -> inst;
        }
        
        while (count < inst -> length) {
            iter2 = emb_model -> vocabdict.find(inst -> words[count]);
            if (iter2 != emb_model -> vocabdict.end()) inst -> ids[count] = iter2 -> second;
            else inst -> ids[count] = -1;
            
            if (fea_params.low_rank) {
                p_inst -> word_ids[p_inst -> count] = inst -> ids[count];
            }
            
            if (fea_params.sum) {
                ostringstream oss;
                oss << "BIAS";
                slot_key = oss.str();
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
            }
            
            if (inst -> head == count) {
                ostringstream oss;
                oss << "HEAD";
                slot_key = oss.str();
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
            }
            
            if (fea_params.postag){// && (count == 3 || count == 6)) {
                ostringstream oss;
                oss << "POS_i_" << inst -> tags[count];
                slot_key = oss.str();
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
                
                oss.str("");
                if (count != 0) oss << "POS_i-1_" << inst -> tags[count - 1];
                else oss << "POS_i-1_BEGIN";
                slot_key = oss.str();
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
                
                oss.str("");
                if (count != inst -> length - 1) oss << "POS_i+1_" << inst -> tags[count + 1];
                else oss << "POS_i+1_END";
                slot_key = oss.str();
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
                
                oss.str("");
                oss << "POS_head_" << inst -> tags[inst -> head];
                slot_key = oss.str();
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
                
                oss.str("");
                oss << "POS_head_" << inst -> tags[inst -> head] << "&POS_i_" << inst -> tags[count];
                slot_key = oss.str();
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
                
                oss.str("");
                if (count != 0) oss << "POS_i-1_" << inst -> tags[count - 1] << "&POS_i_" << inst -> tags[count];
                else oss << "POS_i-1_BEGIN&POS_i_" << inst -> tags[count];
                slot_key = oss.str();
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
                
                oss.str("");
                if (count != inst -> length - 1) oss << "POS_i_" << inst -> tags[count] << "&POS_i+1_" << inst -> tags[count + 1];
                else oss << "POS_i_" << inst -> tags[count] << "&POS_i+1_END";
                slot_key = oss.str();
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
                
            }
            p_inst -> count++;
            count++;
        }
    }
    return 1;
}

void PPDBM2M::ExtractFeatures() {
    word2int::iterator iter2;
    int count = 0;
    int model_id = -1;
    int id;
    string token, tag, slot_key;
    FctDeepModel* p_model = NULL;
    RealFctPathInstance* p_inst = NULL;
    
    if (fea_params.low_rank) {
        slot_key = "PPDB_FCT";
        model_id = SearchDeepFctSlot(slot_key);
        p_model = deep_fct_list[model_id];
        p_inst = p_model -> inst;
    }
    
    while (count < inst -> length) {
        iter2 = emb_model -> vocabdict.find(inst -> words[count]);
        if (iter2 != emb_model -> vocabdict.end()) inst -> ids[count] = iter2 -> second;
        else inst -> ids[count] = -1;
        
        if (fea_params.low_rank) {
            p_inst -> word_ids[p_inst -> count] = inst -> ids[count];
        }
        
        if (fea_params.sum) {
            ostringstream oss;
            oss << "BIAS";
            slot_key = oss.str();
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
        }
        
        if (inst -> head == count) {
            ostringstream oss;
            oss << "HEAD";
            slot_key = oss.str();
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
        }
        
        if (fea_params.postag){// && (count == 3 || count == 6)) {
            ostringstream oss;
            oss << "POS_i_" << inst -> tags[count];
            slot_key = oss.str();
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            
            oss.str("");
            if (count != 0) oss << "POS_i-1_" << inst -> tags[count - 1];
            else oss << "POS_i-1_BEGIN";
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            
            oss.str("");
            if (count != inst -> length - 1) oss << "POS_i+1_" << inst -> tags[count + 1];
            else oss << "POS_i+1_END";
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            
            oss.str("");
            oss << "POS_head_" << inst -> tags[inst -> head];
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            
            oss.str("");
            oss << "POS_head_" << inst -> tags[inst -> head] << "&POS_i_" << inst -> tags[count];
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            
            oss.str("");
            if (count != 0) oss << "POS_i-1_" << inst -> tags[count - 1] << "&POS_i_" << inst -> tags[count];
            else oss << "POS_i-1_BEGIN&POS_i_" << inst -> tags[count];
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            
            oss.str("");
            if (count != inst -> length - 1) oss << "POS_i_" << inst -> tags[count] << "&POS_i+1_" << inst -> tags[count + 1];
            else oss << "POS_i_" << inst -> tags[count] << "&POS_i+1_END";
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            
        }
        p_inst -> count++;
        count++;
    }
}

void PPDBM2M::GetInstanceById(int pos) {
    for (int i = 0; i < deep_fct_list.size(); i++) {
        deep_fct_list[i] -> inst -> Clear();
    }
    string phrase = vocab[pos];
    
    istringstream iss(phrase);
    iss >> inst -> length;
    iss >> inst -> head;
    
    for (int i = 0; i < inst -> length; i++) {
        iss >> inst -> words[i];
    }
    
    for (int i = 0; i < inst -> length; i++) {
        iss >> inst -> tags[i];
    }
    
    for (int i = 0; i < inst -> length; i++) {
        iss >> inst -> clusters[i];
    }
    
    ExtractFeatures();
}

void PPDBM2M::RandomInstance() {
    long id = SampleNegative();
    GetInstanceById((int)id);
}

string PPDBM2M::ToLower(string& s) {
    for (int i = 0; i < s.length(); i++) {
        if (s[i] >= 'A' && s[i] <= 'Z') s[i] += 32;
    }
    return s;
}

int PPDBM2M::AddDeepFctModel(string slot_key) {
    int id;
    feat2int::iterator iter = slot2deep_model.find(slot_key);
    if (iter == slot2deep_model.end()) {
        id = (int)slot2deep_model.size();
        slot2deep_model[slot_key] = id;
        FctDeepModel* p_deep_model = new FctDeepModel(emb_model_list[0]);
        deep_fct_list.push_back(p_deep_model);
        deep_slot_list.push_back(slot_key);
        cout << slot_key << "\t" << id << endl;
        return id;
    }
    return iter -> second;
}

int PPDBM2M::SearchDeepFctSlot(string slot_key) {
    feat2int::iterator iter = slot2deep_model.find(slot_key);
    if (iter == slot2deep_model.end()) return -1;
    else return iter -> second;
}

void PPDBM2M::ForwardProp()
{
    for (int i = 0; i < deep_fct_list.size(); i++) {
        if (deep_fct_list[i] -> inst -> count <= 0) continue;
        deep_fct_list[i] -> ForwardEmb();
    }
}

void PPDBM2M::BackProp()
{
    alpha_old = alpha;
    //    alpha = alpha * ( 1 - eta * lambda );
    real eta_real = eta / alpha;
    //To ADD:
    for (int i = 0; i < deep_fct_list.size(); i++) {
        if (deep_fct_list[i] -> inst -> count <= 0) continue;
        deep_fct_list[i] -> BackProp(b_inst, eta_real);
    }
}

void PPDBM2M::ForwardNCE(ifstream& ifs) {
    double sum;
    int a, c;
    long l2 = 0;
    for (int i = 0; i < num_labels; i++) {
        label_scores[i] = 0.0;
    }
    
    ForwardProp();
    for (int a = 0; a < layer1_size; a++) {
        deep_fct_list[0] -> label_emb[a + l2] = deep_fct_list[0] -> path_emb[a];
    }
    
    for (int j = 1; j <= 15; j++) {
        RandomInstance();
        ForwardProp();
        l2 = j * layer1_size;
        for (int a = 0; a < layer1_size; a++) {
            deep_fct_list[0] -> label_emb[a + l2] = deep_fct_list[0] -> path_emb[a];
        }
    }
    
    LoadInstance(ifs);
    ForwardProp();
    
    sum = 0.0;
    for (a = 0; a < layer1_size; a++) sum += deep_fct_list[0] -> path_emb[a] * deep_fct_list[0] -> label_emb[a];
    b_inst -> scores[0] = sum;
    for (c = 1; c <= 15; c++) {
        sum = 0.0;
        l2 = c * layer1_size;
        for (a = 0; a < layer1_size; a++) sum += deep_fct_list[0] -> path_emb[a] * deep_fct_list[0] -> label_emb[a + l2];
        b_inst -> scores[c] = sum;
    }
    
    float tmp;
    if (b_inst -> scores[0] > MAX_EXP) tmp = exp(MAX_EXP);
    else if (b_inst -> scores[0] < MIN_EXP) tmp = exp(MIN_EXP);
    else tmp = exp(b_inst -> scores[0]);
    b_inst -> scores[0] = tmp;
    sum = b_inst -> scores[0];
    
    for (c = 1; c <= 15; c++) {
        if (b_inst -> scores[c] > MAX_EXP) tmp = exp(MAX_EXP);
        else if (b_inst -> scores[c] < MIN_EXP) tmp = exp(MIN_EXP);
        else tmp = exp(b_inst -> scores[c]);
        b_inst -> scores[c] = tmp;
        sum += b_inst -> scores[c];
    }
    for (c = 0; c < num_labels; c++) {
        b_inst -> scores[c] /= sum;
    }
}

void PPDBM2M::TrainData(string trainfile, string devfile, int type) {
    if (deep_fct_list.size() != 0) {
        printf("L-emb1: %lf\n", deep_fct_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", deep_fct_list[0] -> label_emb[layer1_size]);
        printf("L-emb1: %lf\n", deep_fct_list[0] -> fct_fea_emb[0]);
        printf("L-emb2: %lf\n", deep_fct_list[0] -> fct_fea_emb[layer1_size]);
    }
    
    int count = 0;
    int total = num_inst * iter;
    for (int i = 0; i < iter; i++) {
        cout << "Iter " << i << endl;
        cur_iter = i;
        ifstream ifs(trainfile.c_str());
        //int count = 0;
        while (LoadInstance(ifs)) {
            ForwardNCE(ifs);
            BackProp();
            count++;
            
            if (count % 100 == 0) {
                WeightDecay(eta, lambda);
            }
        }
        if(!adagrad) eta = eta0 * (1 - count / (double)(total + 1));
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();

        if (deep_fct_list.size() != 0) {
            printf("L-emb1: %lf\n", deep_fct_list[0] -> label_emb[0]);
            printf("L-emb2: %lf\n", deep_fct_list[0] -> label_emb[layer1_size]);
            printf("L-emb1: %lf\n", deep_fct_list[0] -> fct_fea_emb[0]);
            printf("L-emb2: %lf\n", deep_fct_list[0] -> fct_fea_emb[layer1_size]);
        }
        EvalLogLoss(trainfile);
        EvalLogLoss(devfile);
        EvalMRR(devfile, THRES1);
        EvalMRR(devfile, THRES2);
    }
    if (deep_fct_list.size() != 0) {
        printf("L-emb1: %lf\n", deep_fct_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", deep_fct_list[0] -> label_emb[layer1_size]);
    }
    
    //ofs.close();
}

void PPDBM2M::EvalLogLoss(string testfile) {
    int total = 0;
    double loss = 0.0;
    next_random = 0;
    
    ifstream ifs(testfile.c_str());
    while (LoadInstance(ifs)) {
        total++;
        ForwardNCE(ifs);
        loss += log(b_inst -> scores[0]);
    }
    cout << loss << endl;
    cout << total << endl;
    cout << "NCE Loss: " << loss / total << endl;
    ifs.close();
}

void PPDBM2M::EvalMRR(string testfile, int threshold)
{
    long long b, c;
    
    int Total = 0;
    double score_total = 0.0;
    long long l1;
    real sim;
    real norm1, norm2;
    real* phrase_norm = (real*)malloc(threshold * sizeof(real));
    real* phrase_emb = (real*)malloc(threshold * layer1_size * sizeof(real));
    real* source_emb = (real*)malloc(layer1_size * sizeof(real));
    
    b = 0;
    for (int i = 0; i< vocab.size(); i++) {
        GetInstanceById(i);
        ForwardProp();
        
        norm1 = 0.0;
        for (c = 0; c < layer1_size; c++) norm1 += deep_fct_list[0] -> path_emb[c] * deep_fct_list[0] -> path_emb[c];
        phrase_norm[b] = norm1;
        
        l1 = b * layer1_size;
        for (c = 0; c < layer1_size; c++) phrase_emb[c + l1] = deep_fct_list[0] -> path_emb[c];
        b++;
        if (b == threshold) break;
    }
    
    ifstream ifs(testfile.c_str());
    while (LoadInstance(ifs)) {
        Total++;
        ForwardProp();
        norm1 = 0.0;
        for (c = 0; c < layer1_size; c++) norm1 += deep_fct_list[0] -> path_emb[c] * deep_fct_list[0] -> path_emb[c];
        for (c = 0; c < layer1_size; c++) source_emb[c] = deep_fct_list[0] -> path_emb[c];
        
        LoadInstance(ifs);
        ForwardProp();
        norm2 = 0.0;
        for (c = 0; c < layer1_size; c++) norm2 += deep_fct_list[0] -> path_emb[c] * deep_fct_list[0] -> path_emb[c];
        
        if (Total % 10 == 0) {
            //printf("%d\r", Total);
            fflush(stdout);
        }
        
        int count_larger = 0;
        real t_sim = 0.0;
        
        for (c = 0; c < layer1_size; c++) t_sim += source_emb[c] * deep_fct_list[0] -> path_emb[c];
        t_sim /= (sqrt (norm1 * norm2));
        for (c = 0; c < layer1_size; c++) source_emb[c] = deep_fct_list[0] -> path_emb[c];
        
        b = 0;
        for (int i = 0; i< vocab.size(); i++) {
            sim = 0.0;
            l1 = b * layer1_size;
            for (c = 0; c < layer1_size; c++) sim += source_emb[c] * phrase_emb[c + l1];
            sim /= (sqrt (phrase_norm[b] * norm2));
            if (sim > t_sim) count_larger++;
            b++;
            if (b == threshold) break;
        }
        
        score_total += (real)1 / (count_larger + 1);
    }
    ifs.close();
    
    vector<string> val;
    printf("\n");
    printf("MRR (threshold %d): %.2f %d %.2f %% \n", b, score_total, Total, score_total / Total * 100);
    free(phrase_emb);
    free(source_emb);
}

void PPDBM2M::PrintModelInfo() {
    cout << "Number of Labels: " << num_labels << endl;
    cout << "Number of Instances: " << num_inst / 2 << endl;
    cout << "Number of Models: " << num_models << endl;
    //    cout << "Number of FCT Slots: " << deep_fct_list[0] -> fct_slotdict.size() << endl;
    cout << "Max length of sentences: " << max_len << endl;
    
    for (int i = 0; i < deep_fct_list.size(); i++) {
        //        cout << "Submodel: " << deep_slot_list[i] << endl;
        //        deep_fct_list[i] -> PrintModelInfo();
    }
}

void PPDBM2M::WeightDecay(real eta_real, real lambda) {
    for (int i = 0; i < deep_fct_list.size(); i++) {
        deep_fct_list[i] -> WeightDecay(eta_real, lambda);
    }
}

void PPDBM2M::InitFreqTable(char* filename) {
    char line_buf[1000];
    string word;
    int id;
    int freq;
    
    ifstream ifs(filename);
    int count = 0;
    ifs.getline(line_buf, 1000, '\n');
    while (strcmp(line_buf, "") != 0) {
        count++;
        ifs.getline(line_buf, 1000, '\n');
    }
    ifs.close();
    
    phrase_vocab.clear();
    vocab.clear();
    freqtable = (int*)malloc(sizeof(int) * count);
    for (int i = 0; i < count; i++) freqtable[i] = 0;
    word2int::iterator iter;
    ifs.open(filename);
    ifs.getline(line_buf, 1000, '\n');
    while (strcmp(line_buf, "") != 0) {
        istringstream iss(line_buf);
        int len;
        int head;
        string key = "";
        iss >> freq;
        iss >> len;
        iss >> head;
        
        ifs.getline(line_buf, 1000, '\n');
        ostringstream oss;
        oss << len;
        oss << " " << head << "\n";
        oss << line_buf << "\n";
        ifs.getline(line_buf, 1000, '\n');
        oss << line_buf;
        key = oss.str();
        
        iter =  phrase_vocab.find(key);
        if (iter == phrase_vocab.end()) {
            id = (int)phrase_vocab.size();
            phrase_vocab[key] = id;
            vocab.push_back(key);
            freqtable[id] = freq;
            //cout << id << " " << freq << endl;
        }
        ifs.getline(line_buf, 1000, '\n');    
    }
    ifs.close();
}

void PPDBM2M::InitUnigramTable() {
    int a, i;
    long long train_words_pow = 0;
    //double train_words_pow = 0.0;
    //real d1, power = 0.75;
    double d1, power = 0.75;
    //double power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    vocab_size = vocab.size();
    for (a = 0; a < vocab_size ; a++) {
        train_words_pow += pow(freqtable[a], power);
    }
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

long PPDBM2M::SampleNegative() {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    long ret = table[(next_random >> 16) % table_size];
    if (ret == 0) ret = next_random % (vocab_size - 1) + 1;
    return ret;
}
