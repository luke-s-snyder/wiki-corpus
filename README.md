# NLP-Project
 
Folders
-------
data: 
- contains full 17 GB wiki corpus for Word2Vec training

sample_data: 
- To create wiki files for LDA training (dictionary, TF-IDF matrix, etc.), use command
    python -m gensim.scripts.make_wikicorpus [INPUT_FILE] [OUTPUT_FILE_PREFIX]
- contains sample ~150 MB wiki corpus for Word2Vec training
- contains trained LDA model and dictionary

lda_filtered_articles:
- contains wiki corpus separated into N files, where each file represents a topic

word2vec_models:
- contains trained word2vec model for each file in lda_filtered_articles

sample_workflow:
- contains example of how to use LDA model on downstream classification set
- evaluates performance of topic-dependent embeddings vs. topic-independent embeddings

src:
- train_lda.py: trains LDA model on wiki corpus
    + USAGE: python train_lda.py [INPUT_FILE_PREFIX] [OUTPUT_FILE] [NUM_TOPICS]
    + EXAMPLE: python train_lda.py ../sample_data/simplewiki ../sample_data/lda.model 100

- process_wiki.py: processes wiki corpus by splitting the corpus into N files for each topic
    + USAGE: python process_wiki.py [LDA_MODEL_PATH] [DICTIONARY_PATH] [INPUT_FILE] [OUTPUT_FILE_PREFIX]
    + EXAMPLE: python process_wiki.py ../sample_data/lda.model ../sample_data/simplewiki_wordids.txt.bz2 ../sample_data/simplewiki-20191101-pages-articles.xml.bz2 ../lda_filtered_articles/simplewiki

- train_word2vec.py: trains word2vec model on each of the N topic-dependent corpora
    + USAGE: python train_word2vec.py [INPUT_FILE_PREFIX] [OUTPUT_FILE_PREFIX] [NUM_TOPICS]
    + EXAMPLE: python train_word2vec.py ../lda_filtered_articles/simplewiki ../word2vec_models/word2vec 100

Project Instructions
---------------------
1. Download wiki dump
2. Run python -m gensim.scripts.make_wikicorpus [INPUT_FILE] [OUTPUT_FILE_PREFIX] to create necessary files for LDA model training
3. Train LDA model with train_lda.py
4. Use LDA model with process_wiki.py to split corpus by major topics
5. Train word2vec model on each topic-dependent corpus with train_word2vec.py
