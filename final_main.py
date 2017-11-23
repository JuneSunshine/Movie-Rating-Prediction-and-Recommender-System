'''
CSCI 630 Final Project: Movie IMDB Score Prediction

@author: Jingyang Li

@instructor: Yuxiao Huang
'''
import pandas as pd
import numpy as np
import final_preparation as fp
import final_learn as fl


def read_and_clean():
    '''
    read in csv file and do some pre-processing
    :return: dataframe
    '''
    df = pd.read_csv("movie_metadata.csv")
    # drop duplicate instances
    df.drop_duplicates(['movie_title'],inplace=True)
    # drop instances with missing values
    df.dropna(subset=['director_name','actor_1_name','actor_2_name','actor_3_name','movie_title'],inplace=True)
    # reset index
    df.reset_index(inplace=True)

    return df

def keywords_weight(df):
    '''
    generate a new feature - plot_keywords_weight
    :param df: original dataframe
    :return: dataframe with new feature
    '''
    kw_count = fp.get_plot_keywords_count(df)

    df2 = df[['plot_keywords','imdb_score']]

    plot_imdb = {}

    keywords = df2.plot_keywords.tolist()

    imdb_score = df2.imdb_score.tolist()

    # iteratively calculate the each keyword's total score
    for kw, ims in zip(keywords, imdb_score):
        if kw is not np.nan:
            for word in kw.split('|'):
                if plot_imdb.get(word):
                    plot_imdb[word] += ims
                else:
                    plot_imdb[word] = ims
    # calculate each keyword's mean score
    for w,s in plot_imdb.items():
        plot_imdb[w] = round(s/ kw_count[w],1)


    df3 = pd.DataFrame.from_dict(plot_imdb,orient='index').reset_index()
    df3.columns = ['keyword','score']
    df3.set_index('keyword',inplace=True)


    df4 = pd.DataFrame.from_dict(kw_count,orient='index').reset_index()
    df4.columns = ['keyword','count']
    df4.set_index('keyword',inplace=True)

    # create a dataframe with these two new attributes
    df5 = df3.join(df4)

    # apply minmax normalization
    maxcount = df5['count'].max()
    mincount = df5['count'].min()
    maxscore = df5['score'].max()
    minscore = df5['score'].min()

    df5['count_weight'] = (df5['count']-mincount)/ (maxcount-mincount)*100
    df5['imdb_weight'] = (df5['score']-minscore)/ (maxscore-minscore)*100
    # here I consider to set weight like this because it seems more reasonable in this way
    df5['weighted_average'] = 0.25*df5['count_weight']+ 0.75 * df5['imdb_weight']
    # generate the weighted average for each keyword
    weighted_avg = df5['weighted_average'].to_dict()

    # remove the '\xa0' character from the movie titles (it causes a lot of trouble)
    df['movie_title'] = df.apply(lambda x: x['movie_title'].replace('\xa0', ''), axis=1)
    movie_titles = df['movie_title'].tolist()

    df.set_index('movie_title',inplace=True)
    plot_keywords_weight = {}

    # for each movie, compute its total keywords weight
    for mt, kw in zip(movie_titles, keywords):
        if kw is not np.nan:
            weight = 0
            for w in kw.split('|'):
                weight += weighted_avg[w]
            num = len(kw.split('|'))
            plot_keywords_weight[mt] = round(weight/num,1)
        else:
            plot_keywords_weight[mt] = 0

    df6 = pd.DataFrame.from_dict(plot_keywords_weight,orient='index')
    df6.columns = ['plot_keywords_weight']

    # add the new feature to original dataframe
    df7 = df.join(df6)
    df7.reset_index(inplace=True)

    return df7



def director_weight(df):
    '''
    generate a new feature - director_weight
    :param df: original dataframe
    :return: dataframe with new feature
    '''
    d_count = fp.get_director_count(df)

    num_instances = df.shape[0]
    df2 = df[['director_name', 'imdb_score']]


    d_imdb = {}

    directors = df2.director_name.tolist()
    imdb_score = df2.imdb_score.tolist()

    # iteratively calculate the total score for each director
    for dr, ims in zip(directors,imdb_score):
        if dr is not np.nan:
            if d_imdb.get(dr):
                d_imdb[dr] += ims
            else:
                d_imdb[dr] = ims

    # calculate the mean score for each director
    for d,s in d_imdb.items():
        d_imdb[d] = round(s / d_count[d],1)

    df3 = pd.DataFrame.from_dict(d_imdb, orient='index').reset_index()
    df3.columns = ['directors','score']
    df3.set_index('directors',inplace=True)

    df4 = pd.DataFrame.from_dict(d_count, orient='index').reset_index()
    df4.columns = ['directors','count']
    df4.set_index('directors',inplace=True)

    # create a dataframe with these two attributes
    df5 = df3.join(df4)

    # apply minmax normalization
    maxcount = df5['count'].max()
    mincount = df5['count'].min()
    maxscore = df5['score'].max()
    minscore = df5['score'].min()

    df5['count_weight'] = (df5['count'] - mincount) / (maxcount - mincount) * 100
    df5['imdb_weight'] = (df5['score'] - minscore) / (maxscore - minscore) * 100

    # apply 1: 3 weight
    df5['weighted_average'] = 0.25*df5['count_weight']+ 0.75 * df5['imdb_weight']

    weighted_avg = df5['weighted_average'].to_dict()

    total_weight = []
    for i in range(num_instances):
        weight = weighted_avg.get(df['director_name'][i])
        total_weight.append(weight)

    # add this feature to original dataframe
    director_weight = pd.DataFrame(total_weight, index=df.index)
    director_weight.columns = ['director_weight']
    df6 = df.join(director_weight)

    return df6


def actor_weight(df):
    '''
    generate a new feature - actor_weight
    :param df: original dataframe
    :return: dataframe with new feature
    '''
    a_count = fp.get_actor_count(df)

    num_instances = df.shape[0]

    df2 = df[['actor_1_name','actor_2_name','actor_3_name','imdb_score']]

    a_imdb = {}
    # in order to pairwisely compare these two attributes
    actors = df2.actor_1_name.tolist() + df2.actor_2_name.tolist() + df2.actor_3_name.tolist()
    imdb_score = df2.imdb_score.tolist() + df2.imdb_score.tolist() + df2.imdb_score.tolist()

    # iteratively calculate the total score for each actor
    for ar, ims in zip(actors,imdb_score):
        if ar is not np.nan:
            if a_imdb.get(ar):
                a_imdb[ar] += ims
            else:
                a_imdb[ar] = ims
    # calculate the mean score for each actor
    for a,s in a_imdb.items():
        a_imdb[a] = round(s / a_count[a],1)


    df3 = pd.DataFrame.from_dict(a_imdb, orient='index').reset_index()
    df3.columns = ['actors', 'score']
    df3.set_index('actors', inplace=True)

    df4 = pd.DataFrame.from_dict(a_count, orient='index').reset_index()
    df4.columns = ['actors', 'count']
    df4.set_index('actors', inplace=True)
    # create a dataframe with these two attributes
    df5 = df3.join(df4)

    # apply minmax normalization
    maxcount = df5['count'].max()
    mincount = df5['count'].min()
    maxscore = df5['score'].max()
    minscore = df5['score'].min()

    df5['count_weight'] = (df5['count'] - mincount) / (maxcount - mincount) * 100
    df5['imdb_weight'] = (df5['score'] - minscore) / (maxscore - minscore) * 100
    # 2:3 weight on actor's appearance and score
    df5['weighted_average'] = 0.4 * df5['count_weight'] + 0.6 * df5['imdb_weight']

    weighted_avg = df5['weighted_average'].to_dict()

    # calculate each movie's actor score
    mean_weight = []
    for i in range(num_instances):
        total = weighted_avg.get(df['actor_1_name'][i]) + weighted_avg.get(df['actor_2_name'][i]) + weighted_avg.get(df['actor_3_name'][i])
        mean = float(total) / 3
        mean_weight.append(mean)

    # add the new feature to original dataframe
    actor_weight = pd.DataFrame(mean_weight, index= df.index)
    actor_weight.columns = ['actor_weight']
    df6 = df.join(actor_weight)

    return df6

def generate_features():
    '''
    generate all the new features and append them to the original dataframe
    :return: dataframe with all new features
    '''
    df = read_and_clean()

    new_df1 = keywords_weight(df)

    new_df2 = director_weight(new_df1)

    new_df3 = actor_weight(new_df2)

    # if any of these new features contains missing values, drop the instance
    for i in range(new_df3.shape[0]):
        if new_df3['plot_keywords_weight'][i] == 0.0 or new_df3['director_weight'][i] == 0.0 or new_df3['actor_weight'][i] == 0.0:
            new_df3 = new_df3.drop(i)

    # reset index after data cleaning
    new_df3.reset_index(inplace=True)
    # compute the mean mixed weight (for linear regression)
    new_df3['mixed_weight'] = (new_df3['plot_keywords_weight'] + new_df3['director_weight'] + new_df3['actor_weight']) / 3

    return new_df3

if __name__ == '__main__':
    print ("Program starts! :P")
    df = generate_features()
    func = input("Please select the method you want to try: \n"
                 "1 for linear regression \n"
                 "2 for Multinomial Logistic Regression \n"
                 "3 for Multi-class Adaboosted Decision Tree \n"
                 "4 for Stochastic Gradient Descent \n"
                 "Press 'Enter' after selection to continue"
                )

    if func == '1':
        fl.linear(df)
    elif func == '2':
        fl.logistic(df)
    elif func == '3':
        fl.dctree(df)
    elif func == '4':
        fl.SGD(df)
    else:
        print ("Do not be naughty! Please enter a valid number")
