from nltk.corpus import brown, stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
from operator import itemgetter 

sentences = brown.sents('ca01')
stopwords = stopwords.words('english')


def textrank(sentences, top_n, stopwords=None):
    """
    sentences = a list of sentences [[s1], [s2], ....]
    top_n = No.of sentences the summary should contain
    stopwords = a list of stopwords
    """
    S = build_similarity_matrix(sentences, stopwords) 
    sentence_ranking = page_rank(S)
 
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranking), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:top_n])
    summary = itemgetter(*selected_sentences)(sentences)
    return summary


def build_similarity_matrix(sentences, stop_words=None):
    S = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                S[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)

    #Normalize the matrix
    for i in range(len(S)):
        S[i] /= S[i].sum()
    
    return S

def page_rank(A, eps=0.0001, d=0.5):
    P = np.ones(len(A)) / len(A)
    while True:
        P_new = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((P_new - P).sum())
        if delta <= eps:
            return P_new
        P = P_new

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # Vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # Vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

n = 3
final_summ = []
for idx, sentence in enumerate(textrank(sentences, n, stopwords)):
    print("%s. %s" % ((idx + 1), ' '.join(sentence)))
    text = ' '.join(sentence)
    final_summ.append(text)

# Summary in 3(value of n) lines
print ('.'.join(final_summ))
