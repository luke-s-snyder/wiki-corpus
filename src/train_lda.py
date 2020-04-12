import gensim
import multiprocessing
import logging
import os.path
import sys

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

inp, outp, num_topics = sys.argv[1:4]

id2word = gensim.corpora.Dictionary.load_from_text(inp + '_wordids.txt.bz2')
mm = gensim.corpora.MmCorpus(inp + '_tfidf.mm')

lda = gensim.models.LdaModel(corpus=mm, id2word=id2word, num_topics=num_topics, update_every=1, passes=1)
lda.save(outp)
