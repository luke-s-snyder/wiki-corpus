import gensim
import os.path
import logging
import pandas as pd
import numpy as np
import sys
from operator import itemgetter

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

inp, outp, lda_model_path, id2word_path, word2vec_path, num_topics = sys.argv[1:7]

df = pd.read_csv(inp)
lda_model = gensim.models.LdaModel.load(lda_model_path)
id2word = gensim.corpora.Dictionary.load_from_text(id2word_path)

topics = []
w2v_embeddings = np.zeros((len(df), 300))

np.random.seed(37)
w2v_rand = np.random.uniform(-0.8, 0.8, 300)

# Compute topic distribution for each tweet.
for i in range(len(df)):
    text = df.loc[i, 'text'].split()
    text = id2word.doc2bow(text)

    topic_probs = lda_model[text]
    if (len(topic_probs) == 0):
        # If text is too short, assign random topic.
        topic = np.random.choice(int(num_topics))

        while (not os.path.isfile(word2vec_path + '_' + str(topic) + '_wv.model')):
            topic = np.random.choice(int(num_topics))
    else:
        # Compute most prevalant topic.
        topic = max(topic_probs, key=itemgetter(1))[0]

        # # If w2v topic model does not exist, get next most prevalant topic.
        # while (not os.path.isfile(word2vec_path + '_' + str(topic[0]) + '_wv.model')):
        #     print(topic_probs)
        #     print(topic)
        #     print('')
        #     topic_probs = list(set(topic_probs) - set(topic))
        #     print(topic_probs)
        #     topic = max(topic_probs, key=itemgetter(1))
        #     print(topic)

    topics.append(topic)

# Compute topic-dependent embeddings for each tweet.
for i in range(int(num_topics)):
    path = word2vec_path + '_' + str(i) + '_wv.model'
    if (not os.path.isfile(path)):
        continue

    w2v_model = gensim.models.KeyedVectors.load(path)

    # Find all tweets with topic i.
    for j, topic in enumerate(topics):
        if topic == i:       
            # Compute average word2vec embeddings.
            average_w2v_embeddings = np.zeros((1, 300))

            text = df.loc[j, 'text'].split()
            for word in text:
                if word in w2v_model:
                    average_w2v_embeddings[0] += w2v_model[word]
                else:
                    average_w2v_embeddings += w2v_rand
            
            average_w2v_embeddings /= len(text)
            w2v_embeddings[j] = average_w2v_embeddings

df['embeddings'] = w2v_embeddings.tolist()
df.to_csv(outp, index=False)
