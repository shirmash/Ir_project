<h1> shir mashiah and ataila solash IR project:</h1>
<h4> bucket name :207437625</h4>

<h1> code structure and organization</h1>

| Pickle File's Name | about the file |
| ------------- | ------------- |
| search_frontend  | containg all main fuction requeired in the project	 |
| search_backend | contaning functinos that are being used in search_frontend  |
| BM25  |containig BM25 class for use in function "search"|
| inverted_index_gcp  |inverted index class |
| creating_inverted_index_gcp |ipynb file that creates all the pikle files in our bucket |
<h1> Pickle Files</h1>
we created the folloing pickle files during our bulilding procees.


all are used in the different function and saved in our bucket:


| Pickle File's Name | about the file |
| ------------- | ------------- |
| PageRank  | 	table of doc's id and there rank score |
| PageView  | table of doc's id their number of views |
| index_text  |inverted index according to the text in wikipidia articels|
| index_title  |inverted index according to the titles in wikipidia articels |
| index_anchor |inverted index according to the anchor in wikipidia articels |


<h1> Retrival Methodes:</h1>

In our search engine we used different retrieval methods:

 Markup :
 * Word2Vec
 
 * BM-25

* TF-IDF

* cosine simularity

* Page View

* Page Rank

<h1> Links </h1>
GCP Bucket : https://console.cloud.google.com/storage/browser/207437625
