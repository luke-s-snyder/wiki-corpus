from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
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

for i in range(int(num_topics)):
    if not os.path.isfile(inp + '_' + str(i) + '.txt'):
        continue

    model = Word2Vec(LineSentence(inp + '_' + str(i) + '.txt'), size=300, window=5, min_count=5,
        workers=multiprocessing.cpu_count())
    model.save(outp + 'word2vec_' + str(i) + '.model')
    model.wv.save(outp + 'word2vec_' + str(i) + '_wv.model')

# model = Word2Vec(LineSentence(inp), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
# model.save(outp)
# model.wv.save('word2vec_models/word2vec_all_wv.model')
