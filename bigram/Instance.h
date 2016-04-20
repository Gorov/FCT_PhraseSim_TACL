//
//  Instance.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-4-16.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_Instance_h
#define PhraseEmb_Instance_h

#include <vector>
#include <sstream>
#include <fstream>
#include <string.h>
#include "EmbeddingModel.h"
using namespace std;

class BaseInstance{
public:
    string word1;
    string word2;
    long id1;
    long id2;
    string tag1;
    string tag2;
    int head1;
    int head2;
};

class Instance: public BaseInstance
{
public:
    /*
    string word1;
    string word2;
    long id1;
    long id2;
    string tag1;
    string tag2;
    int head1;
    int head2;
     */
    vector<string> labels;
    vector<long> ids;
    vector<double> scores;
    
    Instance()
    {
        for (int i = 0; i < 7; i++) {
            labels.push_back("");
        }
        for (int i = 0; i < 7; i++) {
            ids.push_back(0);
        }
        for (int i = 0; i < 7; i++) {
            scores.push_back(0.0);
        }
    }
};

class BinaryInstance: public BaseInstance
{
public:
    /*
    string word1;
    string word2;
    long id1;
    long id2;
    string tag1;
    string tag2;
    int head1;
    int head2;
     */
    string label;
    long label_id;
    double score;
    int positive;
    
    BinaryInstance()
    {
        label = 1;
        score = 0;
    }
};

class SkipgramInstance: public BaseInstance
{
public:
    int num;
    
    vector<string> pos_labels;
    vector<long> pos_ids;
    vector<double> pos_scores;
    
    vector<string> neg_labels;
    vector<long> neg_ids;
    vector<double> neg_scores;
    
    SkipgramInstance()
    {
        pos_labels.resize(10);
        pos_ids.resize(10);
        pos_scores.resize(10);
        neg_labels.resize(150);
        neg_ids.resize(150);
        neg_scores.resize(150);
        /*
        for (int i = 0; i < 7; i++) {
            labels.push_back("");
        }
        for (int i = 0; i < 7; i++) {
            ids.push_back(0);
        }
        for (int i = 0; i < 7; i++) {
            scores.push_back(0.0);
        }
        */
    }
    
    void Clear() {
        pos_labels.clear();
        pos_ids.clear();
        pos_scores.clear();
        neg_labels.clear();
        neg_ids.clear();
        neg_scores.clear();
    }
};

class PPDBB2UInstance: public BaseInstance
{
public:
    int num;
    
    string pos_label;
    long pos_id;
    double pos_score;
    
    vector<string> neg_labels;
    vector<long> neg_ids;
    vector<double> neg_scores;
    
    PPDBB2UInstance()
    {
        neg_labels.resize(15);
        neg_ids.resize(15);
        neg_scores.resize(15);
    }
    
};

class PPDBInstance: public BaseInstance
{
public:
    string target_word1;
    string target_word2;
    string target_tag1;
    string target_tag2;
    long target_id1;
    long target_id2;
    real target_score;
    
    vector<string> neg_labels;
    vector<long> neg_ids;
    vector<double> neg_scores;
    
    vector<string> neg_word1;
    vector<string> neg_word2;
    vector<long> neg_id1;
    vector<long> neg_id2;
    
    PPDBInstance()
    {
        neg_labels.resize(15);
        neg_ids.resize(15);
        neg_scores.resize(15);
        
        neg_word1.resize(15);
        neg_word2.resize(15);
        neg_id1.resize(15);
        neg_id2.resize(15);
    }
    
    void Clear() {
        neg_word1.clear();
        neg_word2.clear();
        neg_id1.clear();
        neg_id2.clear();
        
        neg_labels.clear();
        neg_ids.clear();
        neg_scores.clear();
    }
};

class PPDBTargetInstance: public BaseInstance
{
public:    
    PPDBTargetInstance() {}
    
    void GetValue(PPDBInstance* inst) {
        this -> word1 = inst -> target_word1;
        this -> word2 = inst -> target_word2;
        this -> id1 = inst -> target_id1;
        this -> id2 = inst -> target_id2;
        this -> tag1 = inst -> target_tag1;
        this -> tag2 = inst -> target_tag2;
    }
};

#endif
