//
//  PPDBLearnerB2UDoubleSpace.cpp
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-19.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "PPDBLearnerB2UDoubleSpace.h"

void PPDBLearnerB2UDoubleSpace::EvalMRRDouble(string testfile, int threshold)
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
        
        word2int::iterator iter = emb_model -> vocabdict.find(((PPDBB2UInstance*)inst) -> pos_label );
        if (iter == emb_model -> vocabdict.end()) continue;
        else {
            l1 = iter->second * layer1_size;
            for (c = 0; c < layer1_size; c++) t_sim += emb_model -> syn1neg[c + l1] * emb_p[c];
        }
        
        for (b = 0; b < threshold; b++) {
            if (b == ((PPDBB2UInstance*)inst) -> pos_id) continue;
            l1 = b * layer1_size;
            sim = 0.0;
            for (c = 0; c < layer1_size; c++) sim += (emb_model -> syn1neg[c + l1]) * emb_p[c];
            if (sim > t_sim) count_larger++;
        }
        
        score_total += (real)1 / (count_larger + 1);
    }
    ifs.close();
    
    vector<string> val;
    printf("\n");
    printf("MRR (threshold %d): %.2f %d %.2f %% \n", threshold, score_total, Total, score_total / Total * 100);
}

void PPDBLearnerB2UDoubleSpace::ForwardOutputs()
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
    for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn1neg[a + l1];
    ((PPDBB2UInstance*)inst) -> pos_score = exp(sum);
    
    //negative score
    for (int j = 0; j < 15; j++) {
        sum = 0.0;
        if (((PPDBB2UInstance*)inst)->neg_ids[j] < 0) {
            ((PPDBB2UInstance*)inst) -> neg_scores[j] = 0.0;
            continue;
        }
        l1 = ((PPDBB2UInstance*)inst)->neg_ids[j]  * layer1_size;
        for (a = 0; a < layer1_size; a++) sum += emb_p[a] * emb_model->syn1neg[a + l1];
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

long PPDBLearnerB2UDoubleSpace::BackPropPhrase() {
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
    for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((PPDBB2UInstance*)inst)->pos_score) * emb_model->syn1neg[a + l1];
    if (update_emb) {
        for (a = 0; a < layer1_size; a++) emb_model->syn1neg[a + l1] += eta * (y - ((PPDBB2UInstance*)inst)->pos_score) * emb_p[a];
    }
    
    //negative bp
    for (int j = 0; j < 15; j++) {
        tmpid = ((PPDBB2UInstance*)inst) -> neg_ids[j];
        if (tmpid < 0) {
            continue;
        }
        y = 0;
        l1 = tmpid * layer1_size;
        for (a = 0; a < layer1_size; a++) part_emb_p[a] += (y - ((PPDBB2UInstance*)inst)->neg_scores[j]) * emb_model->syn1neg[a + l1];
        if (update_emb) {
            for (a = 0; a < layer1_size; a++) emb_model->syn1neg[a + l1] += eta * (y - ((PPDBB2UInstance*)inst)->neg_scores[j]) * emb_p[a];
        }
    }
    return 0;
}

void PPDBLearnerB2UDoubleSpace::TrainBigData(string trainfile, string trainsubfile, string devfile) {
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
