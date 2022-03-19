# -*- coding: utf-8 -*-
"""22_SIOP_Workshop.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YqwfUC8XQNaTRjNvjM3eZq_USMI_K0uj

**The latest in Unstructured text analysis AI**

Overview (code porition) 

1. Data  
  *  What data are we using (indeed job descriptions subset) 
  *  Data processing (uploading, stats, etc.)

2. Job Titles
  *  Create embeddings (Sbert | TopicBERT)
  *  Def Query of similarity 
  *  Clustering
  *  topic modeling

3. Job Descriptions
  *  Create embeddings (TopicBert) 
  *  Look at matches with job titles 
  *
"""

# load relivant packages
!pip install -U sentence-transformers
!pip install ipython-autotime

# Commented out IPython magic to ensure Python compatibility.
# bring in relivant models
# explain models and model choice

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import pickle
# %load_ext autotime 

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# import data (indeed Job titles, rather than MH for this> 
#sentences = ['This framework generates embeddings for each input sentence',
#    'Sentences are passed as a list of string.',
#    'The quick brown fox jumps over the lazy dog.']

indeed_title_db = pd.read_csv('/content/drive/MyDrive/indeed_title_output.csv')

indeed_title_db

indeed_job_titles = indeed_title_db['input_title'].head(n =100000)

"""# New Section"""

# make embeddings of titles
indeed_embeddings = model.encode(indeed_job_titles, show_progress_bar = True)
#without gpu and 100k this will take 30 min +, change runtime to gpu... will take 30 sec...

sentence_transformers.util.semantic_search(indeed_embeddings )

# create function pull out similar job titles...

#Store sentences & embeddings on disc
#with open('data/indeed_embeddings.pkl', "wb") as fOut:
#    pickle.dump({'indeed_titles': indeed_titles_only, 'embeddings': indeed_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

# Find similar matches 
#Compute cosine-similarits

#cosine_scores = util.cos_sim(indeed_embeddings, indeed_embeddings)

#Two parameters to tune:
#min_cluster_size: Only consider cluster that have at least 25 elements
#threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(indeed_embeddings, min_community_size=1000, threshold=0.75)

print("Clustering done after {:.2f} sec".format(time.time() - start_time))

#Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", corpus_sentences[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", corpus_sentences[sentence_id])









#The easy way to install Top2Vec is:
!pip install top2vec
#To install pre-trained universal sentence encoder options:

!pip install top2vec[sentence_encoders]
#To install pre-trained BERT sentence transformer options:

!pip install top2vec[sentence_transformers]
#To install indexing options:

!pip install top2vec[indexing]

# To train embeddings from scratch 
from top2vec import Top2Vec

model = Top2Vec(sample)

sample = indeed_title_db.input_title.apply(str).head(n = 1000)

from top2vec import Top2Vec

model = Top2Vec(sample, embedding_model='universal-sentence-encoder')

def onet_semantic_search(top_results, min_unique):
    
    """
    1. Takes in user-inputted job title.
    2. Creates sbert semantic embedding of the title
    3. Returns top matches or top matches with unique SOC matches
    
    User Can:
    
    1. Specify number of results
    2. Minimum number of unique SOC Codes
    
    #add in soc code search and results onet link maybe
    """
    

    input_title = input('Please Enter a Job Title: ')

    print('Performing Semantic Search... \n ====================================== \n ======================================')

    input_embedding = model.encode(input_title, convert_to_numpy =True, show_progress_bar = True)


    top_k = 1000

    #Returns a sorted list with decreasing cosine similarity scores. Entries are dictionaries with the keys ‘corpus_id’ and ‘score’

    results_df = util.semantic_search(input_embedding, onet_embeddings, top_k = top_k)


    #create multiple empty lists in one line
    match_title_index, match_score, match_title = ([] for i in range(3))


    for i in range(0, top_k):
        match_score.append(results_df[0][i]['score'])
        match_title_index.append(results_df[0][i]['corpus_id'])

    df = pd.DataFrame(columns = ['match_title_index', 'match_title', 'match_score'])

    df['match_title_index'] = match_title_index 
    df['match_score'] = match_score
    df['match_score'] = round(df['match_score'], 2)

    for i in range(0,top_k):
       match_title.append(output_title_list[df.iloc[i, 0]])

    df['match_title'] = match_title

    x = df.merge(alternate_titles, how = "left", left_on ='match_title' , right_on ='Alternate Title 2')
    
    if len(x) < top_results:
        print('Not Enough Results to meet query number')
 
    x_filtered_top_results = x[0:top_results]

    if min_unique > 0:
        num_unique = len(pd.unique(x_filtered_top_results['O*NET-SOC Code']))
        if num_unique < min_unique:
            x_unique = x.drop_duplicates(subset =['O*NET-SOC Code'], keep = False)[0:top_results]
            x_unique = x_unique.rename(columns = {'Alternate Title': 'O*NET Alt Title Match', 'Title': 'O*NET Title'})
            x_unique = x_unique[['O*NET Alt Title Match', 'O*NET-SOC Code', 'O*NET Title', 'match_score']]

    is_variable = "x_unique" in locals() or globals()
    
    x_filtered_top_results = x_filtered_top_results.rename(columns = {'Alternate Title': 'O*NET Alt Title Match', 'Title': 'O*NET Title'})
    x_filtered_top_results = x_filtered_top_results[['O*NET Alt Title Match', 'O*NET-SOC Code', 'O*NET Title', 'match_score']]
    
    if is_variable == True:
        print(f' Unique O*NET SOC Code Included in Results Below: \n ========================================== \n ========================================== \n')
        with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           ):  
            return pd.concat([x_filtered_top_results, x_unique])
    else:
        return x_filtered_top_results