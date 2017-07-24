# FCT_PhraseSim_TACL
Tools for phrase similarity used in:

Mo Yu, Mark Dredze. Learning Composition Models for Phrase Embeddings. TACL 2015.

@article{TACL586,
        author = {Mo Yu and Mark Dredze},
        title = {Learning Composition Models for Phrase Embeddings},
        journal = {Transactions of the Association for Computational Linguistics},
        volume = {3},
        year = {2015},
        issn = {2307-387X},
        url =
{https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/586},
        pages = {227--242}
}

We have two code base for bigram experiments and n-gram experiments. The code for n-grams is based on the FCM code at: https://github.com/Gorov/FCM_nips_workshop.

######
#Data#
######

The PPDB XXL data can be found in the data/ directory: each instance consists of three lines:

line 1: word form and POS tag for the first word

line 2: word form and POS tag for the second word

line 3: the target word which paraphrases the above bigram

#################
#Word Embeddings#
#################

The word embeddings can be found at http://yumo.asiteof.me/data/vectors.nyt2011.cbow.bin
