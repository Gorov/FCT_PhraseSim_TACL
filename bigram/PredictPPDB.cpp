
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sstream>

#include "AllLearner.h"

char train_file[MAX_STRING], dev_file[MAX_STRING], output_file[MAX_STRING];
char clus_file[MAX_STRING], baseemb_file[MAX_STRING], freq_file[MAX_STRING];
char model_file[MAX_STRING];
char trainsub_file[MAX_STRING];
char postfix[MAX_STRING];
int iter = 1;
int finetuning = 1;
real alpha = 0.01;

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    output_file[0] = 0;
    if (argc == 1) {
        
        return 0;
    }
    
    
    else{
        int threshold = 10000;
        int withphrase = 0;
        if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-dev", argc, argv)) > 0) strcpy(dev_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-emb", argc, argv)) > 0) strcpy(baseemb_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-clus", argc, argv)) > 0) strcpy(clus_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-freqfile", argc, argv)) > 0) strcpy(freq_file, argv[i + 1]);
        //if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-finetuning", argc, argv)) > 0) finetuning = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
        if ((i = ArgPos((char *)"-model", argc, argv)) > 0) strcpy(model_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-postfix", argc, argv)) > 0) strcpy(postfix, argv[i + 1]);
            
        PPDBLearnerB2U* plearner = new PPDBLearnerB2U(baseemb_file, clus_file, train_file, LM);
        
        plearner -> eta = plearner -> eta0 = 0.01;
        plearner -> iter = 5;
        plearner -> update_emb = true;
        
        cout << "Number of Features: " << plearner -> num_fea << endl;
        cout << "Number of Instances: " << plearner -> fea_model -> num_inst << endl;
        plearner -> InitVocab(freq_file);
        /*
        plearner -> EvalData(train_file);
        plearner -> EvalData(dev_file);
        plearner -> EvalLogLoss(train_file);
        plearner -> EvalLogLoss(dev_file);
        */
        plearner -> EvalMRR(dev_file, 1000);
        plearner -> EvalMRR(dev_file, 5000);
        plearner -> EvalMRR(dev_file, 10000);
        plearner -> EvalMRR(dev_file, 50000);
        plearner -> EvalMRR(dev_file, 100000);
    }
    
    
    return 0;
}
