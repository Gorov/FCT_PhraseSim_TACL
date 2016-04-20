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
