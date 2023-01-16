import struct

from nltk.corpus import stopwords
import re
from inverted_index_gcp import *
import numpy as np
import pandas as pd
import math
from numpy.linalg import norm
import nltk
import gensim.downloader

nltk.download('stopwords')
TUPLE_SIZE = 6
import inverted_index_gcp
import BM25


def download_from_buck(source, dest):
    bucket = storage.Client().get_bucket('207437625')
    blob = bucket.get_blob(source)
    blob.download_to_filename(dest)
    with open(dest, 'rb') as f:
        return pickle.load(f)


def get_inverted_index(source_idx, dest_file, dir_n):
    val = download_from_buck(source_idx, dest_file)
    return inverted_index_gcp.InvertedIndex(dir_n=dir_n).read_index(".", dest_file.split(".")[0])


def get_page_rank_and_view(source, dest):
    bucket = storage.Client().get_bucket('207437625')
    blob = bucket.get_blob(source)
    blob.download_to_filename(dest)
    with open(dest, 'rb') as f:
        return pickle.loads(f.read())


def read_posting_list(inverted, w, dir_n):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, dir_n)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


##pre uploading

##pre load indexes
dir_n_body = "postings_gcp_body"
index_body = get_inverted_index(f"postings_gcp_body/index_text.pkl", f"index_text.pkl", dir_n_body)
dir_n_title = "postings_gcp_title"
index_title = get_inverted_index(f"postings_gcp_title/index_title.pkl", f"index_title.pkl", dir_n_title)
dir_n_anchor = "postings_gcp_anchor"
index_anchor = get_inverted_index(f"postings_gcp_anchor/index_anchor.pkl", f"index_anchor.pkl",
                                  dir_n_anchor)
#pre load page view and page rank
pr = get_page_rank_and_view(f"PageRank/PageRank.pkl", f"PageRank.pkl")
pv = get_page_rank_and_view(f"Page_View/PageView.pkl", f"PageView.pkl")

##calculate max page view and page rank for normalization
max_pr = max(pr.values())
max_pv = max(pv.values())
##trun them to dict
pr = {doc_id: (rank / max_pr) for doc_id, rank in pr.items()}
pv = {doc_id: (view / max_pv) for doc_id, view in pv.items()}

##create body and title bm25 class
bm25_body = BM25.BM25_from_index(index_body, dir_n_body)
bm25_title = BM25.BM25_from_index(index_title, dir_n_title)

##pre load word2vec
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')

#pre load english stopwords
english_stopwords = frozenset(stopwords.words('english'))


def merge_results(text_results, title_results, N):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    title_dict = {doc_id: score for doc_id, score in title_results}
    text_dict = {doc_id: score for doc_id, score in text_results}
    merged_dict = defaultdict()
    for doc_id in text_dict:
        if doc_id in title_dict:
            merged_dict[doc_id] = text_dict[doc_id] * 0.5 + title_dict[doc_id] * 0.3 + pv[doc_id] * 0.1 + pr[
                doc_id] * 0.1
        else:
            merged_dict[doc_id] = text_dict[doc_id] * 0.7 + pv[doc_id] * 0.15 + pr[doc_id] * 0.15
    merged_dict = sorted([(doc_id, score) for doc_id, score in merged_dict.items()], key=lambda x: x[1], reverse=True)[0:N]
    return merged_dict

def tokenize(text):##tokenization
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]
    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = [word for word in tokens if word not in all_stopwords]
    return tokens


def generate_query_tfidf_vector(query_to_search, index):
    """
      Generate a vector representing the query. Each entry within this vector represents a tfidf score.
      The terms representing the query will be the unique terms in the index.

      We will use tfidf on the query as well.
      For calculation of IDF, use log with base 10.
      tf will be normalized based on the length of the query.

      Parameters:
      -----------
      query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                       Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

      index:           inverted index loaded from the corresponding files.

      Returns:
      -----------
      vectorized query with tfidf scores
      """

    epsilon = .0000001
    Q = {}
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]##calculate df
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # calculate idf
            try:
                Q[token] = tf * idf
            except:
                pass
    return Q


def cosine_sim(index, tokenized_query, Q, dir_n, N):
    doc_w_pl_dict = {}
    cosine_dict = {}
    for word in tokenized_query:
        doc_w_pl = read_posting_list(index, word, dir_n)  ##add posting list to dict
        doc_w_pl_dict[word] = doc_w_pl##get posting list
    # Calculate tf_idf for each document
    for word, post in doc_w_pl_dict.items():
        for doc_id, tf in doc_w_pl_dict[word]:##for each doc in posting list
            try:
                tfidf = (tf / index.DL[doc_id]) * math.log((len(index.DL)) / index.df[word])##calculate tfidg
                cosine_dict[doc_id] = cosine_dict.get(doc_id, 0) + np.dot(tfidf, Q[word])
            except:
                pass
    for doc_id in cosine_dict:
        cosine_dict[doc_id] = cosine_dict[doc_id] / (index.norm_dict[doc_id] * norm(list(Q.values())))##calculate final cosine

    return sorted([(doc_id, round(score, 5)) for doc_id, score in cosine_dict.items()], key=lambda x: x[1],
                  reverse=True)[:N]
