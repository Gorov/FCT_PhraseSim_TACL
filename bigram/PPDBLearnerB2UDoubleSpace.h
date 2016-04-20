//
//  PPDBLearnerB2UDoubleSpace.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-19.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_PPDBLearnerB2UDoubleSpace_h
#define PhraseEmb_PPDBLearnerB2UDoubleSpace_h

#include "PPDBLearnerB2U.h"
#include "Commons.h"

class PPDBLearnerB2UDoubleSpace: public PPDBLearnerB2U
{
public:    
    
    PPDBLearnerB2UDoubleSpace();
    ~PPDBLearnerB2UDoubleSpace() {}
    
    PPDBLearnerB2UDoubleSpace(char* embfile):
    PPDBLearnerB2U(embfile) {}
    
    PPDBLearnerB2UDoubleSpace(char* embfile, char* trainfile):
    PPDBLearnerB2U(embfile, trainfile){}
    PPDBLearnerB2UDoubleSpace(char* embfile, char* clusfile, char* trainfile):
    PPDBLearnerB2U(embfile, clusfile, trainfile, EMB){}
    
    PPDBLearnerB2UDoubleSpace(char* embfile, char* clusfile, char* trainfile, int type):
    PPDBLearnerB2U(embfile, clusfile, trainfile, type){}
    
    void TrainBigData(string trainfile, string trainsubfile, string devfile);
    
    void EvalMRRDouble(string testfile, int threshold);
    
    void ForwardOutputs();
    long BackPropPhrase();
};

#endif
