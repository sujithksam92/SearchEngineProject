import re
import pprint
import os  
import json
import numpy as np
import pandas as pd
from pandas import DataFrame
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import pyLDAvis
import pyLDAvis.gensim
from sys import platform
import copy
import matplotlib as plt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use'])

lda_model={}
corpus={}
data={}

topic_json_elem={}
topic_json_dict=[]
# import matplotlib.pyplot as plt
# plt.use('PS')

# plt.show()

def sent_to_words(sent):
    for elem in sent:
        yield(gensim.utils.simple_preprocess(str(elem),deacc=True)) 

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


#Read JSON File
file = 'testdata.json'
with open(file) as train_file:
    # dict_train=json.load(train_file)
    df = pd.read_json(train_file,orient='columns')
    print("Fist Pass (No Modification)")
    # print(df.head(10))
    print("Second Pass (TO LIST)")
    data = df.fc_content.values.tolist()
    # print(*data,sep="\n")
    print("Third Pass (REMOVE EMAILS)")
    
    for item in data:
        # print(*item,sep="\n")
        for elem in item:
            if elem == re.match("\S*@\S*\s?",elem):
                elem=re.sub("\S*@\S*\s?","",elem)
    

    data_words=list(sent_to_words(data))
    # for i in data_words:
    #     print(i)
    #     print("\n")
    # # print(data_words[:1])
    
    #building bigram & trigram
    bigram=gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram=gensim.models.Phrases(bigram[data_words],threshold=100)

    bigram_mod=gensim.models.phrases.Phraser(bigram)
    trigram_mod=gensim.models.phrases.Phraser(trigram)

    # print(trigram_mod[bigram_mod[data_words[0]]])

    #Remove Stopwords
    data_words_nostops = remove_stopwords(data_words)

    #Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # print(data_lemmatized)

    #making Dictionary for LDA
    id2word=corpora.Dictionary(data_lemmatized)

    #making corporus
    texts=data_lemmatized

    #term doc freq
    corpus=[id2word.doc2bow(text) for text in texts]

    # print(corpus)

    #TO SEE THE WORD id2word[INDEX]

    #print([[(id2word[id],freq) for id,freq in cp] for cp in corpus])

    #building LDA Model
    lda_model=gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=20,random_state=100,update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)

    #print keyword in 10 topics
    print(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    #perplexity
    print("Perplexity:",lda_model.log_perplexity(corpus))

    #coherence
    coherence_model_lda=CoherenceModel(model=lda_model,texts=data_lemmatized,dictionary=id2word,coherence='c_v')
    coherence_lda=coherence_model_lda.get_coherence()
    print("Coherence Score:",coherence_lda)
    
    #visualize
    # pyLDAvis.enable_notebook()
    # vis=pyLDAvis.gensim.prepare(lda_model,corpus,id2word)
    # pyLDAvis.save_html(vis,'LDA_Vis.html')

    mallet_path='/Users/sujithsam/Documents/Studies/Stevens/Sem-2/BIS-660-Web-Mining/Research_Engine_Project/mallet-2.0.8/bin/mallet'
    ldamallet=gensim.models.wrappers.LdaMallet(mallet_path,corpus=corpus,num_topics=20, id2word=id2word)
    # Show Topics
    print(ldamallet.show_topics(formatted=False))
    # Show Coherence
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('Coherence Score: ', coherence_ldamallet)
    # data = [re.sub('\S*@\S*\s?','',sent) for sent in data]
    # Can take a long time to run.
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
    # Show graph
    limit=40; start=2; step=6;
    x = range(start, limit, step)
    # plt.plot(x, coherence_values)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.show()
    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
    # Select the model and print the topics
    optimal_model = model_list[5]
    model_topics = optimal_model.show_topics(formatted=False)
    print(optimal_model.print_topics(num_words=10))
    
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    df_dominant_topic.head(10)
    # Group top 5 sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],axis=0)
    
    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    # Show
    print(sent_topics_sorteddf_mallet.head(40))

    # Number of Documents for Each Topic
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)

    # Topic Number and Keywords
    topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

    # Concatenate Column wise
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

    # Show
    print(df_dominant_topics)

    #DF to JSON
    sent_topics_sorteddf_mallet.to_json(r'/Users/sujithsam/Documents/Studies/Stevens/Sem-2/BIS-660-Web-Mining/Research_Engine_Project/sent_topics_sorteddf_mallet.json',orient='index')
    df_dominant_topics.to_json(r'/Users/sujithsam/Documents/Studies/Stevens/Sem-2/BIS-660-Web-Mining/Research_Engine_Project/df_dominant_topics.json',orient='index')










#start LDA
# df = pd.DataFrame.from_dict(json_file_dict)
# print(df.name.unique()) 
# df.head(10)