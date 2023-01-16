import time

import search_backend
from search_backend import *

# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, dir_n,k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = index.DL
        self.N = len(self.DL)
        self.AVGDL = sum(self.DL.values()) / self.N
        self.dir_n = dir_n
        self.term_freq = {}

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {term: math.log(1 + (self.N - self.index.df[term] + 0.5) / (self.index.df[term] + 0.5), 10) for term in
               list_of_tokens if term in self.index.df.keys()}##calculate idf for term
        return idf

    def search(self, query,N=20):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        candidates = self.get_candidate_bm25(query, self.index, self.dir_n)  ##get canidates
        self.idf = self.calc_idf(query)##calculate idf
        scores = sorted([(doc_id, self._score(query, doc_id)) for doc_id in candidates], key=lambda x: x[1],
                        reverse=True)[:N]##calculate score and sort
        return scores

    def get_candidate_bm25(self, tokenized_query, index, dir_n):

        candidates = []
        for term in np.unique(tokenized_query):
            if term in self.index.df.keys() :
                self.term_freq[term] = {}
                doc_w_pl = search_backend.read_posting_list(index, term, dir_n)##get matching posting list
                for doc_id, freq in doc_w_pl :
                    self.term_freq[term][doc_id] = freq##insert tf
                    candidates.append(doc_id)
        return candidates

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        self.idf = self.calc_idf(query)
        score = 0.0
        doc_len = self.DL.get(doc_id,1/0.0000001)
        for term in query:
            if term in self.index.df.keys() :
                if doc_id in self.term_freq[term].keys():
                    freq = self.term_freq[term][doc_id]##get tf
                    numerator = self.idf.get(term,1) * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)

        return score
