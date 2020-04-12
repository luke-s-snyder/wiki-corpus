import gensim
import multiprocessing
import logging
import os.path
import sys
import numpy as np
from operator import itemgetter

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

lda_model_path, id2word_path, inp, outp = sys.argv[1:5]

lda_model = gensim.models.LdaModel.load(lda_model_path)
id2word = gensim.corpora.Dictionary.load_from_text(id2word_path)
wiki = gensim.corpora.WikiCorpus(inp, lemmatize=False, dictionary={})
i = 0

for texts in wiki.get_texts():
    bow = id2word.doc2bow(texts)
    topic_probs = lda_model[bow]

    topic = max(topic_probs, key=itemgetter(1))[0]
    with open(outp + '_' + str(topic) + '.txt', 'a') as f:
        f.write(' '.join(texts) + '\n')

    i += 1
    if (i % 10000 == 0):
        logger.info('Saved '+ str(i) + ' articles.')

logger.info('Finished Saved ' + str(i) + ' articles.')

# with open(outp, 'w') as f: 
#     for text in wiki.get_texts():
#         f.write(' '.join(text) + '\n')

#         i += 1
#         if (i % 10000 == 0):
#             logger.info('Saved '+ str(i) + ' articles.')

# logger.info('Finished Saved ' + str(i) + ' articles.')
