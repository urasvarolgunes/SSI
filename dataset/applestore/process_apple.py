import nltk
from nltk.corpus import stopwords
import numpy as np
stopwords = set(stopwords.words('english'))
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, MinMaxScaler
from collections import defaultdict
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

'''
"id" : App ID
"track_name": App Name
"size_bytes": Size (in Bytes)
"currency": Currency Type #drop
"price": Price amount
"ratingcounttot": User Rating counts (for all version)
"ratingcountver": User Rating counts (for current version) #drop
"user_rating" : Average User Rating value (for all version)
"userratingver": Average User Rating value (for current version) #drop
"ver" : Latest version code #drop
"cont_rating": Content Rating
"prime_genre": Primary Genre
"sup_devices.num": Number of supporting devices
"ipadSc_urls.num": Number of screenshots showed for display
"lang.num": Number of supported languages
"vpp_lic": Vpp Device Based Licensing Enabled
'''

def KNN(X, y, n):
    l = len(y)
    y_hat = []
    for i in range(l):
        X_train = np.delete(X, i, axis = 0)
        y_train = np.delete(y, i, axis = 0)
        neigh = KNeighborsClassifier(n_neighbors = n)
        neigh.fit(X_train, y_train)
        y_hat.extend(neigh.predict(X[i].reshape(1,-1)))
    acc = sum(np.array(y_hat) == y) / l
    return acc

def read_embedding(path):
    with open(path, 'r') as f:
        preembed = f.read().split('\n')
        if len(preembed[-1]) < 2:
            del preembed[-1]
    preembed = [e.split(' ', 1) for e in preembed]
    words = [e[0] for e in preembed]
    embed = [list(map(float, e[1].split())) for e in preembed]
    print(path + ' loaded.')
    
    return dict(zip(words,embed))

def get_avg_embed(descp, stopwords, embed_dict):
    embed_vectors = []
    for word in nltk.word_tokenize(descp):
        if word.isalpha() and word.lower() not in stopwords:
            if word in embed_dict:
                embed_vectors.append(embed_dict[word])
            elif word.lower() in embed_dict:
                embed_vectors.append(embed_dict[word.lower()])
    if not embed_vectors:
        avg_embed = np.zeros(300,)
    else:
        avg_embed = np.mean(embed_vectors, axis = 0)
        
    return avg_embed

def make_google_mat(df, google):
    
    labels = []
    names_in = []
    embeds = []
    for name,label in zip(df.index, df.y):
        if name in google:
            embeds.append(google[name])
            labels.append(label)
            names_in.append(name)

    aff = pd.DataFrame(embeds, index=names_in)
    aff['y'] = labels
    
    print('{} words have pretrained embeds.'.format(len(names_in)))
    return aff

path_dict = {'glove':'glove.840B.300d.txt', 'google':'GoogleNews-vectors-negative300.txt'}

if __name__ == '__main__':
    
    df_base = pd.read_csv('AppleStore.csv').drop(['currency','Unnamed: 0','ver','rating_count_ver','user_rating_ver'], axis=1)
    genre_dict = {genre:i for i,genre in enumerate(df_base['prime_genre'].unique())}
    df_base['prime_genre'] = df_base.prime_genre.map(genre_dict).values

    id_name, y = df_base[['id','track_name']], df_base['prime_genre']
    categorical_feat = pd.DataFrame(OneHotEncoder().fit_transform(df_base[['cont_rating','vpp_lic']]).toarray())

    numeric_feat = df_base.drop(['cont_rating','vpp_lic','id','track_name'], axis=1)
    scaler = MinMaxScaler()
    numeric_feat = pd.DataFrame(scaler.fit_transform(numeric_feat))
    df_base = pd.concat((id_name,numeric_feat, categorical_feat), axis=1)

    for embedding_name in ['google','glove']:

        google = read_embedding(path_dict[embedding_name])

        description_df = pd.read_csv('appleStore_description.csv')
        descriptions = description_df['app_desc']
        embed_features = [get_avg_embed(desc, stopwords, google) for desc in descriptions]
        embed_df = pd.DataFrame(embed_features)
        embed_df['id'] = description_df['id'].values

        df = df_base.merge(embed_df, on = 'id')
        df = df.drop_duplicates(subset=['track_name'])
        df = df.drop('id', axis=1)
        df['y'] = y
        df = df.set_index('track_name')

        googleMat = make_google_mat(df, google)

        googleMat.to_csv('apple_store/{}Mat.csv'.format(embedding_name))

        if not os.path.exists('apple_store/affMat.csv'):
            df.to_csv('apple_store/affMat.csv')

        print(googleMat.shape)
        print(df.shape)
           
        del google
        
        X, y_true = googleMat.iloc[:,:-1].values, googleMat['y'].values.reshape(-1)

        print('base results:')
        for k in [5,10,20]:
            print(k, KNN(X,y_true,k))