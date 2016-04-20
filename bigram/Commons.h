//
//  Commons.h
//  PhraseEmb
//
//  Created by gflfof gflfof on 14-5-4.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef PhraseEmb_Commons_h
#define PhraseEmb_Commons_h

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
//#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

struct word_info {
    char codelen;
    int point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
};

#endif
