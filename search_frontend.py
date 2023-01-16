import json
import time
import inverted_index_gcp
from flask import Flask, request, jsonify
import search_backend
from google.cloud import storage
import requests
import inverted_index_gcp
import re
import BM25

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    dir_n_text = "postings_gcp_body"
    tokenized_query = search_backend.tokenize(query)
    exp_query = tokenized_query.copy()
    t= time.time()
    if len(tokenized_query) < 3 :
        for token in tokenized_query:#expantion of query that is in len of 1 or 2 words
            try:
                exp_for_token= search_backend.glove_vectors.most_similar(token)[:2]##get similar words
                exp_for_token = [tup[0] for tup in exp_for_token if tup[1] >= 0.80]##add word if similarity above 80 precent
                exp_query += exp_for_token
            except:
                pass
    text_results = search_backend.bm25_body.search(exp_query, 50)## calculate BM25 on body
    title_results = search_backend.bm25_title.search(exp_query, 50)## calculate BM25 on title
    merged = search_backend.merge_results(text_results,title_results,20)##merge result
    top_N = [(doc_id, search_backend.index_body.titles[doc_id]) for doc_id, count in merged]##add titles
    return jsonify(top_N)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    dir_n_body = "postings_gcp_body"
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = search_backend.tokenize(query)##tokenize query
    Q = search_backend.generate_query_tfidf_vector(tokenized_query, search_backend.index_body)##calculate query tfidf
    cosine = search_backend.cosine_sim(search_backend.index_body, tokenized_query, Q, dir_n_body, 50)##calculate cosine
    top_N = [(doc_id, search_backend.index_body.titles[doc_id]) for doc_id, count in cosine]##add titles
    # END SOLUTION
    return jsonify(top_N)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    dir_n_title = "postings_gcp_title"
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokenized_query = search_backend.tokenize(query)
    for word in tokenized_query:
        doc_w_pl = search_backend.read_posting_list(search_backend.index_title, word, dir_n_title)##get the word posting list
        res += [posting_tuple[0] for posting_tuple in  doc_w_pl ]##add doc id when word appear in doc
    res = sorted(search_backend.Counter(res).items(), key= lambda x:x[1], reverse = True)##count for each word how many distinct values there are
    res = [(doc_id, search_backend.index_title.titles[doc_id]) for doc_id, count in res]##add titles
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    dir_n_anchor= "postings_gcp_anchor"
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokenized_query = search_backend.tokenize(query)
    for word in tokenized_query:
        doc_w_pl = search_backend.read_posting_list(search_backend.index_anchor, word, dir_n_anchor)##get the word posting list
        res += [posting_tuple[0] for posting_tuple in doc_w_pl]  ##add doc id when word appear in doc
    res = sorted(search_backend.Counter(res).items(), key=lambda x: x[1],
                 reverse=True)  ##count for each word how many distinct values there are
    res = [(doc_id, search_backend.index_anchor.titles[doc_id]) for doc_id, count in res]##get matching titles
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for wiki_id in wiki_ids :
        res.append(search_backend.pr[wiki_id])
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for wiki_id in wiki_ids:
        res.append(search_backend.pv[wiki_id])
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    #run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)