//
//  PrepModel.cpp
//  Preposition_Classification
//
//  Created by gflfof gflfof on 14-12-24.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "PrepModel.h"

void PrepModel::Init(char* embfile, char* trainfile)
{
    fea_params.position = true;
    fea_params.prep = true;
    fea_params.clus = false;
    fea_params.sum = true;
    
    fea_params.low_rank = false;
    
    fea_params.tri_conv = false;
    fea_params.linear = false;//true;
    
    emb_model = new EmbeddingModel(embfile);
    layer1_size = (int)emb_model -> layer1_size;
    emb_model_list.push_back(emb_model);
    
    BuildModelsFromData(trainfile);
    //    InitFeatureModel((int)layer1_size, trainfile, type);
    num_models = (int)(coarse_fct_list.size() + deep_fct_list.size() + convolution_fct_list.size());
    num_labels = (int)labeldict.size();
    inst -> scores.resize(num_labels);
    
    //    max_frame_slots = fea_model -> max_len - 1;
    //    if (fea_model -> max_len == 0) max_frame_slots = 1;
    alpha = 1.0;
    lambda = 0.0;
}

void PrepModel::InitSubmodels() {
    for (int i = 0; i < coarse_fct_list.size(); i++) {
        coarse_fct_list[i] -> num_labels = num_labels;
        coarse_fct_list[i] -> InitModel();
        coarse_fct_list[i] -> update_emb = update_emb;
        coarse_fct_list[i] -> adagrad = adagrad;
    }
    for (int i = 0; i < deep_fct_list.size(); i++) {
        deep_fct_list[i] -> num_labels = num_labels;
        deep_fct_list[i] -> InitModel();
        deep_fct_list[i] -> update_emb = update_emb;
        deep_fct_list[i] -> adagrad = adagrad;
    }
    for (int i = 0; i < convolution_fct_list.size(); i++) {
        convolution_fct_list[i] -> num_labels = num_labels;
        convolution_fct_list[i] -> InitModel();
        convolution_fct_list[i] -> update_emb = update_emb;
        convolution_fct_list[i] -> adagrad = adagrad;
    }
}

void PrepModel::BuildModelsFromData(char* trainfile) {
    layer1_size = (int)emb_model_list[0] -> layer1_size;
    
    ifstream ifs(trainfile);
    num_inst = 0;
    while (LoadInstanceInit(ifs)) {
        num_inst++;
    }
    
    labellist.resize(labeldict.size());
    for (feat2int::iterator iter = labeldict.begin(); iter != labeldict.end(); iter++) {
        labellist[iter -> second] = iter -> first;
    }
    ifs.close();
}

int PrepModel::LoadInstanceInit(ifstream &ifs) {
    int id, model_id;
    int beg1 = 0, end1 = 0, beg2 = 0, end2 = 0;
    char line_buf[5000], line_buf2[1000];
    vector<int> trigram_id;
    trigram_id.resize(3);
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    
    {
        istringstream iss(line_buf);
        iss >> inst -> label;
        feat2int::iterator iter = labeldict.find(inst -> label);
        if (iter == labeldict.end()) {
            id = (int)labeldict.size();
            labeldict[inst -> label] = id;
        }
        
        if (fea_params.prep) {
            iss >> ((PrepInstance*)inst) -> prep_word;
        }
    }
    {
        ifs.getline(line_buf, 5000, '\n');
        istringstream iss2(line_buf);
        int count = 0;
        int model_id = -1;
        string token, tag, slot_key;
        
        while (iss2 >> token) {
            ToLower(token);
            inst -> words[count] = token;
            
            if (fea_params.low_rank) {
                slot_key = "PREP_FCT";
                model_id = AddDeepFctModel(slot_key);
            }
            
            if (fea_params.sum) {
                ostringstream oss;
                oss << "BIAS";
                slot_key = oss.str();
                if (fea_params.low_rank) {
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
                else {
                    AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                }
            }
            
            if (fea_params.position){// && (count == 3 || count == 6)) {
                ostringstream oss;
                oss << "POS_" << count;
                slot_key = oss.str();
                if (fea_params.low_rank) {
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
                else {
                    AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                }
                
                if (fea_params.prep) {
                    oss.str("");
                    oss << "PREP_" << ((PrepInstance*)inst) -> prep_word;
                    slot_key = oss.str();
                    if (fea_params.low_rank) {
                        deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    }
                    else {
                        AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                    }
                    
                    oss.str("");
                    oss << "POS_PREP_" << count << "_" << ((PrepInstance*)inst) -> prep_word;
                    slot_key = oss.str();
                    if (fea_params.low_rank) {
                        deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    }
                    else {
                        AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                    }
                }
            }
            count++;
        }
        
        inst -> len = count + 1;
        if (inst -> len > max_len) {max_len = inst -> len;}
    }
    
    return 1;
}

int PrepModel::LoadInstance(ifstream& ifs, int type) {
    return LoadInstance(ifs);
}

int PrepModel::LoadInstance(ifstream &ifs) {
    word2int::iterator iter2;
    char line_buf[5000];
    vector<int> trigram_id;
    trigram_id.resize(3);
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    
    for (int i = 0; i < deep_fct_list.size(); i++) {
        deep_fct_list[i] -> inst -> Clear();
    }
    for (int i = 0; i < coarse_fct_list.size(); i++) {
        coarse_fct_list[i] -> inst -> Clear();
    }
    for (int i = 0; i < convolution_fct_list.size(); i++) {
        convolution_fct_list[i] -> inst -> Clear();
    }
    
    {
        istringstream iss(line_buf);
        iss >> inst -> label;
        feat2int::iterator iter = labeldict.find(inst -> label);
        if (iter == labeldict.end()) inst -> label_id = -1;
        else inst -> label_id = iter -> second;
        
        if (fea_params.prep) {
            iss >> ((PrepInstance*)inst) -> prep_word;
        }
    }
    {
        ifs.getline(line_buf, 5000, '\n');
        
        istringstream iss2(line_buf);
        int count = 0;
        int model_id = -1;
        int id;
        string token, tag, slot_key;
        FctDeepModel* p_model = NULL;
        RealFctPathInstance* p_inst = NULL;
        
        if (fea_params.low_rank) {
            slot_key = "PREP_FCT";
            model_id = SearchDeepFctSlot(slot_key);
            p_model = deep_fct_list[model_id];
            p_inst = p_model -> inst;
        }
        
        while (iss2 >> token) {
            ToLower(token);
            inst -> words[count] = token;
            iter2 = emb_model -> vocabdict.find(token);
            if (iter2 != emb_model -> vocabdict.end()) inst -> word_ids[count] = iter2 -> second;
            else inst -> word_ids[count] = -1;
            
            if (fea_params.low_rank) {
                p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
            }
            
            if (fea_params.sum) {
                ostringstream oss;
                oss << "BIAS";
                slot_key = oss.str();
                if (fea_params.low_rank) {
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                }
                else {
                    AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                }
            }
            
            if (fea_params.position){// && (count == 3 || count == 6)) {
                ostringstream oss;
                oss << "POS_" << count; //bug here
                slot_key = oss.str();
                AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                
                if (fea_params.prep) {
                    oss.str("");
                    oss << "PREP_" << ((PrepInstance*)inst) -> prep_word;
                    slot_key = oss.str();
                    if (fea_params.low_rank) {
                        id = p_model -> SearchRealFCTSlot(slot_key);
                        p_inst -> PushFctFea(id, p_inst -> count);
                    }
                    else {
                        AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                    }
                    
                    oss.str("");
                    oss << "POS_PREP_" << count << "_" << ((PrepInstance*)inst) -> prep_word;
                    slot_key = oss.str();
                    if (fea_params.low_rank) {
                        id = p_model -> SearchRealFCTSlot(slot_key);
                        p_inst -> PushFctFea(id, p_inst -> count);
                    }
                    else {
                        AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                    }
                }
            }
            if (fea_params.low_rank) {
                p_inst -> count++;
            }
            count++;
        }
        inst -> len = count + 1;
    }
    return 1;
}

void PrepModel::PushWordFeature(string slot_key, int wordid, FctDeepModel* p_model, RealFctPathInstance* p_inst) {
    int id;
    if (fea_params.low_rank) {
        id = p_model -> SearchRealFCTSlot(slot_key);
        p_inst -> PushFctFea(id, p_inst -> count);
    }
    else {
        AddCoarseFctModel2List(slot_key, wordid, false);
    }
}

int PrepModel::AddCoarseFctModel2List(string slot_key, int word_id, bool add) {
    int model_id, id;
    string fea_key;
    if (add) {
        model_id = AddDeepFctModel(slot_key);
        fea_key = "FCT_bias";
        deep_fct_list[model_id] -> AddRealFCTSlot(fea_key);
        return model_id;
    }
    else {
        model_id = SearchDeepFctSlot(slot_key);
        if (model_id >= 0) {
            FctDeepModel* p_model = deep_fct_list[model_id];
            RealFctPathInstance* p_inst = p_model -> inst;
            p_inst -> word_ids[p_inst -> count] = word_id;
            fea_key = "FCT_bias";
            id = p_model -> SearchRealFCTSlot(fea_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            p_inst -> count++;
        }
        return model_id;
    }
}

int PrepModel::AddConvolutionModel2List(string slot_key, vector<int> word_id_vec, bool add) {
    int model_id;
    string fea_key;
    if (add) {
        model_id = AddConvolutionModel(slot_key, (int)word_id_vec.size());
        return model_id;
    }
    else {
        model_id = SearchConvolutionSlot(slot_key);
        if (model_id >= 0) {
            FctConvolutionModel* p_model = convolution_fct_list[model_id];
            FctConvolutionInstance* p_inst = p_model -> inst;
            for (int i = 0; i < word_id_vec.size(); i++) {
                p_inst -> ngram_ids[p_inst -> num_ngram][i] = word_id_vec[i];
            }
            p_inst -> num_ngram++;
        }
        return model_id;
    }
}

string PrepModel::ToLower(string& s) {
    for (int i = 0; i < s.length(); i++) {
        if (s[i] >= 'A' && s[i] <= 'Z') s[i] += 32;
    }
    return s;
}

int PrepModel::AddCoarseFctModel(string slot_key) {
    return -1;
}

int PrepModel::AddDeepFctModel(string slot_key) {
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

int PrepModel::AddConvolutionModel(string slot_key, int length) {
    int id;
    feat2int::iterator iter = slot2convolution_model.find(slot_key);
    if (iter == slot2convolution_model.end()) {
        id = (int)slot2convolution_model.size();
        slot2convolution_model[slot_key] = id;
        FctConvolutionModel* p_conv_model = new FctConvolutionModel(emb_model_list[0], length);
        convolution_fct_list.push_back(p_conv_model);
        convolution_slot_list.push_back(slot_key);
        cout << slot_key << "\t" << id << endl;
        return id;
    }
    return iter -> second;
}

int PrepModel::SearchCoarseFctSlot(string slot_key) {
    feat2int::iterator iter = slot2coarse_model.find(slot_key);
    if (iter == slot2coarse_model.end()) return -1;
    else return iter -> second;
}

int PrepModel::SearchDeepFctSlot(string slot_key) {
    feat2int::iterator iter = slot2deep_model.find(slot_key);
    if (iter == slot2deep_model.end()) return -1;
    else return iter -> second;
}

int PrepModel::SearchConvolutionSlot(string slot_key) {
    feat2int::iterator iter = slot2convolution_model.find(slot_key);
    if (iter == slot2convolution_model.end()) return -1;
    else return iter -> second;
}

void PrepModel::ForwardProp()
{
    real sum;
    int c;
    for (int i = 0; i < num_labels; i++) {
        inst -> scores[i] = 0.0;
    }
    for (int i = 0; i < coarse_fct_list.size(); i++) {
        coarse_fct_list[i] -> ForwardProp(inst);
    }
    for (int i = 0; i < deep_fct_list.size(); i++) {
        if (deep_fct_list[i] -> inst -> count <= 0) continue;
        deep_fct_list[i] -> ForwardProp(inst);
    }
    for (int i = 0; i < convolution_fct_list.size(); i++) {
        if (convolution_fct_list[i] -> inst -> num_ngram <= 0) continue;
        convolution_fct_list[i] -> ForwardProp(inst);
    }
    sum = 0.0;
    for (c = 0; c < num_labels; c++) {
        float tmp;
        if (inst -> scores[c] > MAX_EXP) tmp = exp(MAX_EXP);
        else if (inst -> scores[c] < MIN_EXP) tmp = exp(MIN_EXP);
        else tmp = exp(inst -> scores[c]);
        inst -> scores[c] = tmp;
        sum += inst -> scores[c];
        
    }
    for (c = 0; c < num_labels; c++) {
        inst -> scores[c] /= sum;
    }
}

void PrepModel::BackProp()
{
    alpha_old = alpha;
    //    alpha = alpha * ( 1 - eta * lambda );
    real eta_real = eta / alpha;
    for (int i = 0; i < coarse_fct_list.size(); i++) {
        coarse_fct_list[i] -> BackProp(inst, eta_real);
    }
    for (int i = 0; i < deep_fct_list.size(); i++) {
        if (deep_fct_list[i] -> inst -> count <= 0) continue;
        deep_fct_list[i] -> BackProp(inst, eta_real);
    }
    for (int i = 0; i < convolution_fct_list.size(); i++) {
        if (convolution_fct_list[i] -> inst -> num_ngram <= 0) continue;
        convolution_fct_list[i] -> BackProp(inst, eta_real);
    }
}

void PrepModel::TrainData(string trainfile, string devfile, int type) {
    if (deep_fct_list.size() != 0) {
        printf("L-emb1: %lf\n", deep_fct_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", deep_fct_list[0] -> label_emb[layer1_size]);
        printf("L-emb1: %lf\n", deep_fct_list[0] -> fct_fea_emb[0]);
        printf("L-emb2: %lf\n", deep_fct_list[0] -> fct_fea_emb[layer1_size]);
    }
    if(convolution_fct_list.size() != 0) {
        printf("L-emb1: %lf\n", convolution_fct_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", convolution_fct_list[0] -> label_emb[layer1_size]);
        printf("L-emb1: %lf\n", convolution_fct_list[0] -> W0[0]);
        printf("L-emb2: %lf\n", convolution_fct_list[0] -> W0[layer1_size]);
    }
    int count = 0;
    int total = num_inst * iter;
    for (int i = 0; i < iter; i++) {
        cout << "Iter " << i << endl;
        cur_iter = i;
        ifstream ifs(trainfile.c_str());
        //int count = 0;
        while (LoadInstance(ifs, type)) {
            ForwardProp();
            BackProp();
            count++;
            
            if (count % 100 == 0) {
                WeightDecay(eta, lambda);
            }
        }
        if(!adagrad) eta = eta0 * (1 - count / (double)(total + 1));
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
        //if(i >= 5) for (int a = 0; a < num_labels * num_slots * layer1_size; a++) label_emb[a] = (1 -lambda_prox) * label_emb[a];
        //if(i >= 5) if(update_emb) for (int a = 0; a < emb_model->vocab_size * layer1_size; a++) emb_model -> syn0[a] = (1 - lambda_prox) * emb_model -> syn0[a];
        if (deep_fct_list.size() != 0) {
            printf("L-emb1: %lf\n", deep_fct_list[0] -> label_emb[0]);
            printf("L-emb2: %lf\n", deep_fct_list[0] -> label_emb[layer1_size]);
            printf("L-emb1: %lf\n", deep_fct_list[0] -> fct_fea_emb[0]);
            printf("L-emb2: %lf\n", deep_fct_list[0] -> fct_fea_emb[layer1_size]);
        }
        if(convolution_fct_list.size() != 0) {
            printf("L-emb1: %lf\n", convolution_fct_list[0] -> label_emb[0]);
            printf("L-emb2: %lf\n", convolution_fct_list[0] -> label_emb[layer1_size]);
            printf("L-emb1: %lf\n", convolution_fct_list[0] -> W0[0]);
            printf("L-emb2: %lf\n", convolution_fct_list[0] -> W0[layer1_size]);
        }
        EvalData(trainfile, type);
        EvalData(devfile, type);
    }
    if (deep_fct_list.size() != 0) {
        printf("L-emb1: %lf\n", deep_fct_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", deep_fct_list[0] -> label_emb[layer1_size]);
    }
    if(convolution_fct_list.size() != 0) {
        printf("L-emb1: %lf\n", convolution_fct_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", convolution_fct_list[0] -> label_emb[layer1_size]);
    }
    
    //ofs.close();
}

void PrepModel::EvalData(string trainfile, int type) {
    int total = 0;
    int right = 0;
    int positive = 0;
    int tp = 0;
    int pos_pred = 0;
    double max, max_p;
    real prec, rec;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs, type)) {
        if (inst -> label_id == -1) {
            continue;
        }
        //continue;
        total++;
        if (inst -> label_id != 0) positive++;
        ForwardProp();
        max = -1;
        max_p = -1;
        for (int i = 0; i < num_labels; i++){
            if (inst -> scores[i] > max) {
                max = inst -> scores[i];
                max_p = i;
            }
        }
        if (max_p != 0) pos_pred ++;
        if (max_p == inst -> label_id) {
            right++;
            if (inst -> label_id != 0) tp++;
        }
    }
//    cout << right << " " << tp << " "  << positive << " " << pos_pred << endl;
    cout << "Acc: " << (float)right / total << endl;
//    rec = (real)tp / positive;
//    prec = (real)tp / pos_pred;
//    cout << "Prec: " << prec << endl;
//    cout << "Rec:" << rec << endl;
//    real f1 = 2 * prec * rec / (prec + rec);
//    cout << "F1:" << f1 << endl;
//    cout << std::setprecision(4) << prec * 100 << "\t" << rec * 100 << "\t" << f1 * 100 << endl;
    ifs.close();
}

//void PrepModel::EvalData(string trainfile, string outfile, int type) {
//    int total = 0;
//    int right = 0;
//    int positive = 0;
//    int tp = 0;
//    int pos_pred = 0;
//    int id = 8000;
//    double max, max_p;
//    real prec, rec;
//    ifstream ifs(trainfile.c_str());
//    ofstream ofs(outfile.c_str());
//    while (LoadInstance(ifs, type)) {
//        if (inst -> label_id == -1) {
//            continue;
//        }
//        //continue;
//        total++;
//        if (inst -> label_id != 0) positive++;
//        ForwardProp();
//        max = -1;
//        max_p = -1;
//        for (int i = 0; i < num_labels; i++){
//            if (inst -> scores[i] > max) {
//                max = inst -> scores[i];
//                max_p = i;
//            }
//        }
//        ofs << (id + total) << "\t" << labellist[max_p] << endl;// << "\t" << inst -> scores[max_p] << endl;
//        if (max_p != 0) pos_pred ++;
//        if (max_p == inst -> label_id) {
//            right++;
//            if (inst -> label_id != 0) tp++;
//        }
//    }
//    cout << right << " " << tp << " "  << positive << " " << pos_pred << endl;
//    cout << "Acc: " << (float)right / total << endl;
//    rec = (real)tp / positive;
//    prec = (real)tp / pos_pred;
//    cout << "Prec: " << prec << endl;
//    cout << "Rec:" << rec << endl;
//    real f1 = 2 * prec * rec / (prec + rec);
//    cout << "F1:" << f1 << endl;
//    cout << std::setprecision(4) << prec * 100 << "\t" << rec * 100 << "\t" << f1 * 100 << endl;
//    ifs.close();
//    ofs.close();
//}

void PrepModel::PrintModelInfo() {
    cout << "Number of Labels: " << num_labels << endl;
    cout << "Number of Instances: " << num_inst << endl;
    cout << "Number of Models: " << num_models << endl;
    //    cout << "Number of FCT Slots: " << deep_fct_list[0] -> fct_slotdict.size() << endl;
    cout << "Max length of sentences: " << max_len << endl;
    
    for (int i = 0; i < deep_fct_list.size(); i++) {
        //        cout << "Submodel: " << deep_slot_list[i] << endl;
        //        deep_fct_list[i] -> PrintModelInfo();
    }
    
    for (int i = 0; i < coarse_fct_list.size(); i++) {
        //        cout << "Submodel: " << coarse_slot_list[i] << endl;
    }
    
}

void PrepModel::WeightDecay(real eta_real, real lambda) {
    for (int i = 0; i < deep_fct_list.size(); i++) {
        deep_fct_list[i] -> WeightDecay(eta_real, lambda);
    }
}

