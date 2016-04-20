//
//  PPDBInstance.h
//  Preposition_Classification
//
//  Created by gflfof gflfof on 15-1-14.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef Preposition_Classification_PPDBInstance_h
#define Preposition_Classification_PPDBInstance_h

#include <vector>
using namespace std;

class PPDBNgramInstance {
public:
    vector<string> words;
    vector<string> tags;
    vector<string> clusters;
    vector<int> ids;
    
    vector<double> scores;
    
    int head;
    int length;
    
//    vector<string> target_words;
//    vector<string> target_tags;
//    vector<int> target_ids;
//    
//    int target_length;
    
    PPDBNgramInstance()
    {
        words.resize(15);
        tags.resize(15);
        clusters.resize(15);
        ids.resize(15);
        scores.resize(15 + 1);
        length = 0;
//        target_words.resize(15);
//        target_tags.resize(15);
//        target_ids.resize(15);
    }
    
    void Clear() {
        words.clear();
        tags.clear();
        clusters.clear();
        ids.clear();
        
//        target_words.clear();
//        target_tags.clear();
//        target_ids.clear();
        
        length = 0;
//        target_length = 0;
    }
};

#endif
