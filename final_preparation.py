'''
CSCI 630 Final Project: Movie IMDB Score Prediction

@author: Jingyang Li

@instructor: Yuxiao Huang
'''
from collections import Counter

def get_plot_keywords_count(df):
    '''
    count keywords appearance
    :param df: dataframe
    :return: count
    '''
    cnt = Counter()
    plot_keywords = df['plot_keywords'].dropna()
    # count all the keywords in each movie
    for words in plot_keywords:
        l = words.split('|')
        cnt.update(l)
    return cnt

def get_director_count(df):
    '''
    count director appearance
    :param df: dataframe
    :return: count
    '''
    directors = df['director_name'].dropna()
    cnt = Counter(directors)

    return cnt

def get_actor_count(df):
    '''
    count actor appearance
    :param df: dataframe
    :return: count
    '''
    actors = df[['actor_1_name','actor_2_name','actor_3_name']].dropna()
    cnt = Counter()
    # iteratively count actor 1,2,3
    for i in range(3):
        cnt.update(actors.iloc[:,i])
    return cnt

