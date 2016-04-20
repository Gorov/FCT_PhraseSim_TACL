//
//  FeatureModel.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-4-14.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "FeatureModel.h"
#include "EmbeddingModel.h"
#include <stdio.h>

void FeatureModel::InitFeatDict(char *filename) {
    int i;
    Instance* inst = new Instance();
    ifstream ifs(filename);
    while (LoadInstance(ifs, inst)) {
        CollectFeaBNP(inst);
    }
    num_fea = feadict.size();
//    param = (double*)malloc(num_fea * dim * sizeof(double));
//    b1s = (double*)malloc(dim * sizeof(double));
//    b2s = (double*)malloc(dim * sizeof(double));
    param = (real*)malloc(num_fea * dim * sizeof(real));
    b1s = (real*)malloc(dim * sizeof(real));
    b2s = (real*)malloc(dim * sizeof(real));
    for (i = 0; i < num_fea * dim; i++) param[i] = 0.0;
    //b1 = 1.0;
    //b2 = 1.0;
    for (i = 0; i < dim; i++) b1s[i] = 1.0;
    for (i = 0; i < dim; i++) b2s[i] = 1.0;
    //b2 = 1.5;
    ifs.close();
}

void FeatureModel::InitFeatDictBinary(char *filename) {
    int i;
    BinaryInstance* inst = new BinaryInstance();
    ifstream ifs(filename);
    while (LoadInstance(ifs, inst)) {
        CollectFeaBNP(inst);
    }
    num_fea = feadict.size();
//    param = (double*)malloc(num_fea * dim * sizeof(double));
//    b1s = (double*)malloc(dim * sizeof(double));
//    b2s = (double*)malloc(dim * sizeof(double));
    param = (real*)malloc(num_fea * dim * sizeof(real));
    b1s = (real*)malloc(dim * sizeof(real));
    b2s = (real*)malloc(dim * sizeof(real));
    for (i = 0; i < num_fea * dim; i++) param[i] = 0.0;
    //b1 = 1.0;
    //b2 = 1.0;
    for (i = 0; i < dim; i++) b1s[i] = 1.0;
    for (i = 0; i < dim; i++) b2s[i] = 1.0;
    //b2 = 1.5;
    ifs.close();
}

void FeatureModel::InitFeatDictSkipgram(char *filename) {
    int i;
    SkipgramInstance* inst = new SkipgramInstance();
    ifstream ifs(filename);
    num_inst = 0;
    while (LoadInstance(ifs, inst)) {
        CollectFeaBNP(inst);
        num_inst++;
    }
    num_fea = feadict.size();
//    param = (double*)malloc(num_fea * dim * sizeof(double));
//    b1s = (double*)malloc(dim * sizeof(double));
//    b2s = (double*)malloc(dim * sizeof(double));
    param = (real*)malloc(num_fea * dim * sizeof(real));
    b1s = (real*)malloc(dim * sizeof(real));
    b2s = (real*)malloc(dim * sizeof(real));
    for (i = 0; i < num_fea * dim; i++) param[i] = 0.0;
    for (i = 0; i < dim; i++) b1s[i] = 1.0;
    for (i = 0; i < dim; i++) b2s[i] = 1.0;
    //b2 = 1.5;
    ifs.close();
}

void FeatureModel::InitFeatDict(char *filename, int type) {
    int i;
    if (type == PPDB_INST) {
        PPDBInstance* inst = new PPDBInstance();
        PPDBTargetInstance* target_inst = new PPDBTargetInstance();
        ifstream ifs(filename);
        num_inst = 0;
        while (LoadInstance(ifs, inst)) {
            target_inst -> GetValue(inst);
            CollectFeaBNP(inst);
            CollectFeaBNP(target_inst);
            num_inst++;
        }
        ifs.close();
    }
    num_fea = feadict.size();
    //    param = (double*)malloc(num_fea * dim * sizeof(double));
    //    b1s = (double*)malloc(dim * sizeof(double));
    //    b2s = (double*)malloc(dim * sizeof(double));
    param = (real*)malloc(num_fea * dim * sizeof(real));
    b1s = (real*)malloc(dim * sizeof(real));
    b2s = (real*)malloc(dim * sizeof(real));
    for (i = 0; i < num_fea * dim; i++) param[i] = 0.0;
    for (i = 0; i < dim; i++) b1s[i] = 1.0;
    for (i = 0; i < dim; i++) b2s[i] = 1.0;
    //b2 = 1.5;
}

int FeatureModel::AddFeatDict(char *filename, int obj_type) {
    if (obj_type == BINARY_INST) {
        BinaryInstance* inst = new BinaryInstance();
        ifstream ifs(filename);
        int count = 0;
        while (LoadInstance(ifs, inst)) {
            CollectFeaBNP(inst);
            count ++;
        }
        ifs.close();
        return count;
    }
    if (obj_type == PPDB_INST || obj_type == SKIPGRAM_INST) {
        SkipgramInstance* inst = new SkipgramInstance();
        ifstream ifs(filename);
        int count = 0;
        while (LoadInstance(ifs, inst)) {
            CollectFeaBNP(inst);
            count++;
        }
        ifs.close();
        return count;
    }
}

void FeatureModel::InitFeatPara() {
    int i;
    num_fea = feadict.size();
    param = (real*)malloc(num_fea * dim * sizeof(real));
    b1s = (real*)malloc(dim * sizeof(real));
    b2s = (real*)malloc(dim * sizeof(real));
    for (i = 0; i < num_fea * dim; i++) param[i] = 0.0;
    for (i = 0; i < dim; i++) b1s[i] = 1.0;
    for (i = 0; i < dim; i++) b2s[i] = 1.0;
}

void FeatureModel::InitAddtionalFeatPara() {
    int i;
    int c;
    int old_num_fea = num_fea;
    num_fea = feadict.size();
    real* new_param = (real*)malloc(num_fea * dim * sizeof(real));
    real* new_b1s = (real*)malloc(dim * sizeof(real));
    real* new_b2s = (real*)malloc(dim * sizeof(real));
    
    for (c = 0; c < dim; c++) {
        memcpy(&new_param[c * num_fea], &param[c * old_num_fea], old_num_fea * sizeof(real));
        for (i = old_num_fea; i < num_fea; i++) new_param[c * num_fea + i] = 0.0;
    }
    
    memcpy(new_b1s, b1s, dim * sizeof(real));
    memcpy(new_b2s, b2s, dim * sizeof(real));
    free(param);
    free(b1s);
    free(b2s);
    
    param = new_param;
    b1s = new_b1s;
    b2s = new_b2s;
}

void FeatureModel::InitClusDict(char *filename) {
    int i;
    ifstream ifs(filename);
    char tmp[1000];
    ifs.getline(tmp, 1000, '\n');
    while (strcmp(tmp, "") != 0 ) {
        istringstream iss(tmp);
        string word; iss >> word;
        string clus; iss >> clus;
        clusdict[word] = clus;
        ifs.getline(tmp, 1000, '\n');
    }
    ifs.close();
}

int FeatureModel::LoadInstance(ifstream &ifs, SkipgramInstance* inst) {
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
    
    ifs.getline(line_buf, 1000, '\n');
    return 1;
}

int FeatureModel::LoadInstance(ifstream &ifs, Instance* inst) {
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
        inst -> labels[i] = line_buf;
    }
    return 1;
}

int FeatureModel::LoadInstance(ifstream &ifs, BinaryInstance* inst) {
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
    ifs.getline(line_buf, 1000, '\n');
    iss.clear();
    iss.str(line_buf);
    iss >> inst -> label;
    ifs.getline(line_buf, 1000, '\n');
    iss.clear();
    iss.str(line_buf);
    iss >> inst -> positive;
    
    return 1;
}

int FeatureModel::LoadInstance(ifstream &ifs, PPDBInstance* inst) {
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
    iss >> ((PPDBInstance*)inst) -> target_word1;
    iss >> ((PPDBInstance*)inst) -> target_tag1;
    ifs.getline(line_buf, 1000, '\n');
    iss.clear();
    iss.str(line_buf);
    iss >> ((PPDBInstance*)inst) -> target_word2;
    iss >> ((PPDBInstance*)inst) -> target_tag2;
    
    return 1;
}

void FeatureModel::ExtractFeaBNP(Instance *inst, int* feat_vec1, int* feat_vec2, int* counts)
{
    feat2int::iterator iter;
    word2clus::iterator iter2;
    word2clus::iterator iter3;
    // word1
    counts[0] = 0;
    string feat = "1FEAT_t_i+" + inst -> tag1;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec1[counts[0]] = iter -> second;
        counts[0] ++;
    }
    feat = "1FEAT_t_i+1+" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec1[counts[0]] = iter -> second;
        counts[0] ++;
    }
    feat = "1FEAT_t_i_i+1+" + inst -> tag1 + "_" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec1[counts[0]] = iter -> second;
        counts[0] ++;
    }
    feat = "1FEAT_ishead=0";
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec1[counts[0]] = iter -> second;
        counts[0] ++;
    }
    feat = "1FEAT_headtag+" + inst->tag2;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec1[counts[0]] = iter -> second;
        counts[0] ++;
    }
    
    if (lex_fea) {
        iter2 = clusdict.find(inst -> word1);
        if (iter2 != clusdict.end()) {
            feat = "1FEAT_c_i+" + iter2 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec1[counts[0]] = iter -> second;
                counts[0] ++;
            }
        }
        iter3 = clusdict.find(inst -> word2);
        if (iter3 != clusdict.end()) {
            feat = "1FEAT_c_i+1+" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec1[counts[0]] = iter -> second;
                counts[0] ++;
            }
            feat = "1FEAT_headclus+" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec1[counts[0]] = iter -> second;
                counts[0] ++;
            }
        }
        if (iter3 != clusdict.end() && iter2 != clusdict.end()) {
            feat = "1FEAT_c_i_i+1+" + iter2->second + "_" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec1[counts[0]] = iter -> second;
                counts[0] ++;
            }
        }
    }
    
    // word2
    counts[1] = 0;
    feat = "2FEAT_t_i+" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec2[counts[1]] = iter -> second;
        counts[1]++;
    }
    feat = "2FEAT_t_i-1+" + inst -> tag1;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec2[counts[1]] = iter -> second;
        counts[1] ++;
    }
    feat = "2FEAT_t_i-1_i+" + inst -> tag1 + "_" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec2[counts[1]] = iter -> second;
        counts[1] ++;
    }
    feat = "2FEAT_ishead=1";
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec2[counts[1]] = iter -> second;
        counts[1] ++;
    }
    
    if (lex_fea) {
        iter2 = clusdict.find(inst -> word2);
        if (iter2 != clusdict.end()) {
            feat = "2FEAT_c_i+" + iter2 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec2[counts[1]] = iter -> second;
                counts[1] ++;
            }
        }
        iter3 = clusdict.find(inst -> word1);
        if (iter3 != clusdict.end()) {
            feat = "2FEAT_c_i-1+" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec2[counts[1]] = iter -> second;
                counts[1] ++;
            }
        }
        if (iter3 != clusdict.end() && iter2 != clusdict.end()) {
            feat = "2FEAT_c_i-1_i+" + iter3->second + "_" + iter2 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec2[counts[1]] = iter -> second;
                counts[1] ++;
            }
        }
    }
}

//void FeatureModel::CollectFeaBNP(Instance *inst)
//{
//    feat2int::iterator iter;
//    word2clus::iterator iter2, iter3;
//    // word1
//    string feat = "1FEAT_t_i+" + inst -> tag1;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "1FEAT_t_i+1+" + inst -> tag2;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "1FEAT_t_i_i+1+" + inst -> tag1 + "_" + inst -> tag2;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "1FEAT_ishead=0";
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "1FEAT_headtag+" + inst->tag2;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    if (lex_fea) {
//        iter2 = clusdict.find(inst -> word1);
//        if (iter2 != clusdict.end()) {
//            feat = "1FEAT_c_i+" + iter2 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//        iter3 = clusdict.find(inst -> word2);
//        if (iter3 != clusdict.end()) {
//            feat = "1FEAT_c_i+1+" + iter3 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//            feat = "1FEAT_headclus+" + iter3 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//        if (iter3 != clusdict.end() && iter2 != clusdict.end()) {
//            feat = "1FEAT_c_i_i+1+" + iter2->second + "_" + iter3 -> second;
//            iter = feadict.find(feat);
//            if (iter != feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//    }
//    
//    // word2
//    feat = "2FEAT_t_i+" + inst -> tag2;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "2FEAT_t_i-1+" + inst -> tag1;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "2FEAT_t_i-1_i+" + inst -> tag1 + "_" + inst -> tag2;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "2FEAT_ishead=1";
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    if (lex_fea) {
//        iter2 = clusdict.find(inst -> word2);
//        if (iter2 != clusdict.end()) {
//            feat = "2FEAT_c_i+" + iter2 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//        iter3 = clusdict.find(inst -> word1);
//        if (iter3 != clusdict.end()) {
//            feat = "2FEAT_c_i-1+" + iter3 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//        if (iter3 != clusdict.end() && iter2 != clusdict.end()) {
//            feat = "2FEAT_c_i-1_i+" + iter3->second + "_" + iter2 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//    }
//}

void FeatureModel::ExtractFeaBNP(BaseInstance *inst, int* feat_vec1, int* feat_vec2, int* counts)
{
    feat2int::iterator iter;
    word2clus::iterator iter2;
    word2clus::iterator iter3;
    // word1
    counts[0] = 0;
    string feat = "1FEAT_t_i+" + inst -> tag1;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec1[counts[0]] = iter -> second;
        counts[0] ++;
    }
    feat = "1FEAT_t_i+1+" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec1[counts[0]] = iter -> second;
        counts[0] ++;
    }
    feat = "1FEAT_t_i_i+1+" + inst -> tag1 + "_" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec1[counts[0]] = iter -> second;
        counts[0] ++;
    }
    feat = "1FEAT_ishead=0";
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec1[counts[0]] = iter -> second;
        counts[0] ++;
    }
    feat = "1FEAT_headtag+" + inst->tag2;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec1[counts[0]] = iter -> second;
        counts[0] ++;
    }
    
    if (lex_fea) {
        iter2 = clusdict.find(inst -> word1);
        if (iter2 != clusdict.end()) {
            feat = "1FEAT_c_i+" + iter2 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec1[counts[0]] = iter -> second;
                counts[0] ++;
            }
        }
        iter3 = clusdict.find(inst -> word2);
        if (iter3 != clusdict.end()) {
            feat = "1FEAT_c_i+1+" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec1[counts[0]] = iter -> second;
                counts[0] ++;
            }
            feat = "1FEAT_headclus+" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec1[counts[0]] = iter -> second;
                counts[0] ++;
            }
        }
        if (iter3 != clusdict.end() && iter2 != clusdict.end()) {
            feat = "1FEAT_c_i_i+1+" + iter2->second + "_" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec1[counts[0]] = iter -> second;
                counts[0] ++;
            }
        }
    }
    
    // word2
    counts[1] = 0;
    feat = "2FEAT_t_i+" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec2[counts[1]] = iter -> second;
        counts[1]++;
    }
    feat = "2FEAT_t_i-1+" + inst -> tag1;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec2[counts[1]] = iter -> second;
        counts[1] ++;
    }
    feat = "2FEAT_t_i-1_i+" + inst -> tag1 + "_" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec2[counts[1]] = iter -> second;
        counts[1] ++;
    }
    feat = "2FEAT_ishead=1";
    iter = feadict.find(feat);
    if (iter != feadict.end()) {
        feat_vec2[counts[1]] = iter -> second;
        counts[1] ++;
    }
    
    if (lex_fea) {
        iter2 = clusdict.find(inst -> word2);
        if (iter2 != clusdict.end()) {
            feat = "2FEAT_c_i+" + iter2 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec2[counts[1]] = iter -> second;
                counts[1] ++;
            }
        }
        iter3 = clusdict.find(inst -> word1);
        if (iter3 != clusdict.end()) {
            feat = "2FEAT_c_i-1+" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec2[counts[1]] = iter -> second;
                counts[1] ++;
            }
        }
        if (iter3 != clusdict.end() && iter2 != clusdict.end()) {
            feat = "2FEAT_c_i-1_i+" + iter3->second + "_" + iter2 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                feat_vec2[counts[1]] = iter -> second;
                counts[1] ++;
            }
        }
    }
}

//void FeatureModel::CollectFeaBNP(BaseInstance *inst)
//{
//    feat2int::iterator iter;
//    word2clus::iterator iter2, iter3;
//    // word1
//    string feat = "1FEAT_t_i+" + inst -> tag1;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "1FEAT_t_i+1+" + inst -> tag2;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "1FEAT_t_i_i+1+" + inst -> tag1 + "_" + inst -> tag2;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "1FEAT_ishead=0";
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "1FEAT_headtag+" + inst->tag2;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    if (lex_fea) {
//        iter2 = clusdict.find(inst -> word1);
//        if (iter2 != clusdict.end()) {
//            feat = "1FEAT_c_i+" + iter2 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//        iter3 = clusdict.find(inst -> word2);
//        if (iter3 != clusdict.end()) {
//            feat = "1FEAT_c_i+1+" + iter3 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//            feat = "1FEAT_headclus+" + iter3 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//        if (iter3 != clusdict.end() && iter2 != clusdict.end()) {
//            feat = "1FEAT_c_i_i+1+" + iter2->second + "_" + iter3 -> second;
//            iter = feadict.find(feat);
//            if (iter != feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//    }
//    
//    // word2
//    feat = "2FEAT_t_i+" + inst -> tag2;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "2FEAT_t_i-1+" + inst -> tag1;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "2FEAT_t_i-1_i+" + inst -> tag1 + "_" + inst -> tag2;
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    feat = "2FEAT_ishead=1";
//    iter = feadict.find(feat);
//    if (iter == feadict.end()) {
//        feadict[feat] = (int)feadict.size();
//    }
//    if (lex_fea) {
//        iter2 = clusdict.find(inst -> word2);
//        if (iter2 != clusdict.end()) {
//            feat = "2FEAT_c_i+" + iter2 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//        iter3 = clusdict.find(inst -> word1);
//        if (iter3 != clusdict.end()) {
//            feat = "2FEAT_c_i-1+" + iter3 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//        if (iter3 != clusdict.end() && iter2 != clusdict.end()) {
//            feat = "2FEAT_c_i-1_i+" + iter3->second + "_" + iter2 -> second;
//            iter = feadict.find(feat);
//            if (iter == feadict.end()) {
//                feadict[feat] = (int)feadict.size();
//            }
//        }
//    }
//}

void FeatureModel::CollectFeaBNP(BaseInstance *inst)
{
    feat2int::iterator iter;
    word2clus::iterator iter2, iter3;
    int val;
    // word1
    string feat = "1FEAT_t_i+" + inst -> tag1;
    iter = feadict.find(feat);
    if (iter == feadict.end()) {
        val = (int)feadict.size();
        feadict[feat] = val;
    }
    feat = "1FEAT_t_i+1+" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter == feadict.end()) {
        val = (int)feadict.size();
        feadict[feat] = val;
    }
    feat = "1FEAT_t_i_i+1+" + inst -> tag1 + "_" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter == feadict.end()) {
        val = (int)feadict.size();
        feadict[feat] = val;
    }
    feat = "1FEAT_ishead=0";
    iter = feadict.find(feat);
    if (iter == feadict.end()) {
        val = (int)feadict.size();
        feadict[feat] = val;
    }
    feat = "1FEAT_headtag+" + inst->tag2;
    iter = feadict.find(feat);
    if (iter == feadict.end()) {
        val = (int)feadict.size();
        feadict[feat] = val;
    }
    if (lex_fea) {
        iter2 = clusdict.find(inst -> word1);
        if (iter2 != clusdict.end()) {
            feat = "1FEAT_c_i+" + iter2 -> second;
            iter = feadict.find(feat);
            if (iter == feadict.end()) {
                val = (int)feadict.size();
                feadict[feat] = val;
            }
        }
        iter3 = clusdict.find(inst -> word2);
        if (iter3 != clusdict.end()) {
            feat = "1FEAT_c_i+1+" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter == feadict.end()) {
                val = (int)feadict.size();
                feadict[feat] = val;
            }
            feat = "1FEAT_headclus+" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter == feadict.end()) {
                val = (int)feadict.size();
                feadict[feat] = val;
            }
        }
        if (iter3 != clusdict.end() && iter2 != clusdict.end()) {
            feat = "1FEAT_c_i_i+1+" + iter2->second + "_" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter != feadict.end()) {
                val = (int)feadict.size();
                feadict[feat] = val;
            }
        }
    }
    
    // word2
    feat = "2FEAT_t_i+" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter == feadict.end()) {
        val = (int)feadict.size();
        feadict[feat] = val;
    }
    feat = "2FEAT_t_i-1+" + inst -> tag1;
    iter = feadict.find(feat);
    if (iter == feadict.end()) {
        val = (int)feadict.size();
        feadict[feat] = val;
    }
    feat = "2FEAT_t_i-1_i+" + inst -> tag1 + "_" + inst -> tag2;
    iter = feadict.find(feat);
    if (iter == feadict.end()) {
        val = (int)feadict.size();
        feadict[feat] = val;
    }
    feat = "2FEAT_ishead=1";
    iter = feadict.find(feat);
    if (iter == feadict.end()) {
        val = (int)feadict.size();
        feadict[feat] = val;
    }
    if (lex_fea) {
        iter2 = clusdict.find(inst -> word2);
        if (iter2 != clusdict.end()) {
            feat = "2FEAT_c_i+" + iter2 -> second;
            iter = feadict.find(feat);
            if (iter == feadict.end()) {
                val = (int)feadict.size();
                feadict[feat] = val;
            }
        }
        iter3 = clusdict.find(inst -> word1);
        if (iter3 != clusdict.end()) {
            feat = "2FEAT_c_i-1+" + iter3 -> second;
            iter = feadict.find(feat);
            if (iter == feadict.end()) {
                val = (int)feadict.size();
                feadict[feat] = val;
            }
        }
        if (iter3 != clusdict.end() && iter2 != clusdict.end()) {
            feat = "2FEAT_c_i-1_i+" + iter3->second + "_" + iter2 -> second;
            iter = feadict.find(feat);
            if (iter == feadict.end()) {
                val = (int)feadict.size();
                feadict[feat] = val;
            }
        }
    }
}


void FeatureModel::SaveModel(string modelfile) {
    //code for save cluster and load cluster
    FILE* fileout = fopen(modelfile.c_str(), "wb");
    fwrite(&num_fea, sizeof(unsigned long), 1, fileout);
    fwrite(&dim, sizeof(int), 1, fileout);
    for (feat2int::iterator iter = feadict.begin(); iter != feadict.end(); iter++) {
        fprintf(fileout, "%s %d\n", iter->first.c_str(), iter->second);
    }
    fwrite(param, sizeof(real), num_fea * dim, fileout);
    fwrite(b1s, sizeof(real), dim, fileout);
    fwrite(b2s, sizeof(real), dim, fileout);
    fclose(fileout);
}

void FeatureModel::LoadModel(string modelfile) {
    char feature[100];
    int value;
    string key;
    FILE* filein = fopen(modelfile.c_str(), "rb");
    feadict.clear();
    fread(&num_fea, sizeof(unsigned long), 1, filein);
    fread(&dim, sizeof(int), 1, filein);
    for (int i = 0; i < num_fea; i++) {
        fscanf(filein, "%s %d\n", feature, &value);
        key = feature;
        feadict[key] = value;
    }
    param = (real*)malloc(num_fea * dim * sizeof(real));
    b1s = (real*)malloc(dim * sizeof(real));
    b2s = (real*)malloc(dim * sizeof(real));
    
    fread(param, sizeof(real), num_fea * dim, filein);
    fread(b1s, sizeof(real), dim, filein);
    fread(b2s, sizeof(real), dim, filein);
    fclose(filein);
}

