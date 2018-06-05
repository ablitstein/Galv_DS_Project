import pymysql as pm
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import TweetTokenizer
import nltk.data
import numpy as np
import pandas as pd
from  sklearn.ensemble import RandomForestClassifier as rf
import string
import re
from collections import Counter
from os import path
from PIL import Image
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from random import randint
from flask import Flask, render_template,request, url_for,flash,redirect
#from models import Album
#from app import app
# from db_setup import init_db, db_session
#from forms import MusicSearchForm
from yelp_scrub import YelpScrub
app = Flask(__name__)

#PEOPLE_FOLDER = os.path.join('static', 'people_photo')
#app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

ys = YelpScrub(1)
def qr_db():
    s = ''' SELECT
        id, name, neighborhood, city, state, address
        FROM yelp_db.business;
        '''
    return s

def run_gen(id):
    '''
    creates 6 wordclouds, 3 positive, 3 negative, for 1-3 grams
    using the YelpScrub class and methods

    Args:
        id: the id of the business being processed

    Returns:
        creates jpg files pos1,pos2,pos3,neg1,neg2,neg3
    '''

    #ref_id = randint(1000000, 9999999)
    #doc_list = []
    ys = YelpScrub()
    (dfH,dfL)= ys.run2(id)

    dhl = list(dfH['cText'])
    high_full_list = ys.create_df_list(dhl,3)

    dll = list(dfL['cText'])
    low_full_list = ys.create_df_list(dll,3)
    low_len = len(low_full_list)
    high_len = len(high_full_list)
    name = ys.get_name()

    high_rev, low_rev = ys.rep_rev(dfH, dfL)


    for i in range(low_len):
        fs = ys.find_shared(low_full_list[i],high_full_list[i],10)
        ys.show_wordcloud(' '.join(ys.clean_list(low_full_list[i],fs)),size = 4, name = 'static/neg{}'.format(i+1),makefile = True,show = False)
        #doc_list.append('static/{}neg{}.jpg'.format(ref_id,i+1))

    for j in range(high_len):
        fs = ys.find_shared(low_full_list[j],high_full_list[j],10)
        ys.show_wordcloud(' '.join(ys.clean_list(high_full_list[j],fs)),size = 4, name = 'static/pos{}'.format(j+1),makefile = True,show = False)
        #doc_list.append('static/{}neg{}.jpg'.format(ref_id,j+1))
    #return doc_list
    return name, high_rev, low_rev

# home page
#@app.route('/web/')
#def index():
#    return render_template('index.html', title='Hello!')

@app.route('/')
def index():
    return render_template('splash.html', bus_info = 'Big Bad Business')



@app.route('/index')
def show_index():
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg')
    return render_template('index.html', user_image = 'test.jpg')


@app.route('/hello')
def api_articles():
    return 'hello world!'

@app.route('/yelp/<yid>')
def api_yelpest(yid):

    nameL,high_revs,low_revs = run_gen(yid)
    name = "{}  {}, {}".format(nameL[1],nameL[2],nameL[3])
    return render_template('index.html',bus_info = name,
                           pos1 = '/static/pos1.jpg',pos2 = '/static/pos2.jpg',pos3 = '/static/pos3.jpg',
                           neg1 = '/static/neg1.jpg', neg2 ='/static/neg2.jpg',neg3 = '/static/neg3.jpg',
                           hr1=str(high_revs[0]),hr2=str(high_revs[1]),
                           lr1=str(low_revs[0]),lr2=str(low_revs[1]))




#"http://0.0.0.0:8000/yelp/1dMU2kz5AhTC6N1W9xwuQ"


#@app.route('/more/')
#def more():
#    return render_template('starter_template.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
