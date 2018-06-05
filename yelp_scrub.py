import pymysql as pm
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
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

class YelpScrub():

    def __init__(self, ngram=1):
        self.ngram = ngram
        self.cred = 'cred.csv'
        self.db_user = 'X'
        self.db_pw = 'X'
        self.word_list = []
        self.businfo = []

    def get_cred(self,file_name):
        '''
        Gets DB Credentials from file

        Args:
            file_name: the name of the file containing db credentials.

        Returns:
            A tuple with the username and pw
        '''

        cred_df = pd.read_csv(file_name)

        return (cred_df.columns[0],cred_df.columns[1])


    def find_bus(self,id,user,pw):
        '''
        Pulls information from database

        Args:
            id: a string containing the business ID
            user: username for the database
            pw: username for the database

        Returns:
            A List of information from the database in cluding:

            review.id, business.id, business.name, business.city,
            business.state, review.text, review.stars, review.date,
            review.useful, review.funny, review.cool
        '''

        # Open database connection, create cursor
        db = pm.connect("localhost",user,pw,"yelp_db" )
        cursor = db.cursor()

        #populate query
        query1 ='''
        SELECT
            review.id,
            business.id,
            business.name,
            business.city,
            business.state,
            review.text,
            review.stars,
            review.date,
            review.useful,
            review.funny,
            review.cool
        FROM
            yelp_db.business AS business
                LEFT JOIN
            yelp_db.review AS review ON (review.business_id = business.id)
        WHERE
            business.id = '{}';
        '''.format(id)

        # execute SQL query using execute() method.
        cursor.execute(query1)

        # Fetch data
        data = cursor.fetchall()
        db.close()

        return list(data)

    def dat_to_list(self,data):
        '''
        Puts inputed data into list form

        Args:
            data: data to be put into a list

        Returns:
            A List form of the data
        '''

        d=[]
        for dat in data:
            d.append(list(dat))
        return d


    def no_p(self,my_str):
        '''
        removes punctuation from the string

        Args:
            my_str: the string to be cleansed of punctuation

        Returns:
            A String without punctuation
        '''
        # define punctuation
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        no_punct = ""
        for char in my_str:
           if char not in punctuations:
               no_punct = no_punct + char

        return(no_punct)


    def clean_text(self,data):
        '''
        Cleans text and returns the text that has been
        tokenized, lemmatiezed, and scrubbed of stopwords

        Args:
            data: the string to be tokenized

        Returns:
            A String with tokenized lemmatized, and cleaned text
        '''

        #tokenize data by breaking up words into lists:
        data_tokenized =[]
        for d in data:
            tok =  TweetTokenizer()
            data_tokenized.append(tok.tokenize(d))
        #[TweetTokenizer(content.lower()) for content in data]
       # data_tokenized = punct_scrub(data_tokenized)

        #lemmatize words using WordNet:
        wordnet = WordNetLemmatizer()
        #snowball = SnowballStemmer('english')

        data_wordnet = [[wordnet.lemmatize(word) for word in words] for words in data_tokenized]
        #data_Snowball = [[snowball.stem(word) for word in words] for words in data_docs]

         #remove stop_words from the lists of words:
        stopW = set(stopwords.words('english'))
        stopW.add('wa')
        data_docs = []
        for words in data_wordnet:
            s = ''
            for word in words:
                if 'www' not in word:
                    if word.lower() not in stopW:
                        s+=(self.no_p(word).lower())+' '
            data_docs.append(s[0:-2].replace('  ',' '))

        w = []
        for word_string in data_docs:
           w.append(str(word_string).lower())

        df3 = []
        for d in w:
            df3.append(self.no_p(d))

        return df3

    def make_ngram(self,num_grams,text):
        '''
        Creates ngrams from a string

        Args:
            num_grams: The first parameter.
            text: A multi-word string.

        Returns:
            A list of strings, of the N grams attached by a '_'
        '''
        ngrams = lambda a, n: zip(*[a[i:] for i in range(n)])

        grams = ngrams(text.split(), num_grams)

        l_grams = list(grams)
        joined_grams = []

        for ng in l_grams:
            joined_grams.append('_'.join(list(ng)))

        return joined_grams

    def create_df_list(self,data_list,max_ngram):
        '''
        creats a list of all words and ngrams for the given dataset

        Args:
            data_list: the data to be itereated trhough, processed and dumped into the list
            max_ngram: the largest number of ngrams to include

        Returns:
            A list of strings for all words in all the text strings
        '''
        d_list = []

        for i in range(max_ngram):
            d_n = []
            for data_string in data_list:
                d_n+= self.make_ngram(i+1,data_string)
            d_list.append(d_n)
        return d_list

    def find_shared(self,low_list, high_list,num_common = 7):
        '''
        Finds common words between highs and lows to remove

        Args:
            low_list = list of low rated words
            high_list = list of high rated words
            num_common = the number of commonalities to  be compaired

        Returns:
            a list of words to be scrubbed because of shared usage
        '''

        lc = list(dict(Counter(low_list).most_common(num_common)).keys())
        hc = list(dict(Counter(high_list).most_common(num_common)).keys())

        idd = []

        for a in lc:
             if a in hc:
                idd.append(a)

        return idd

    def clean_list(self,word_list,s_words):
        '''
        removes items from a list

        Args:
            low_list = list of low rated words
            high_list = list of high rated words
            num_common = the number of commonalities to  be compaired

        Returns:
            a list of words to be scrubbed because of shared usage
        '''
        return [x for x in word_list if x not in s_words]

    def show_wordcloud(self,data, title = 'High',size = 5,name= 'wc',file = 'X', makefile = False, show = True):

        stopwords = set(STOPWORDS)
        stopwords.add('shoe')

        if file != 'X':
            masking = np.array(Image.open(X))

            wordcloud = WordCloud(
                background_color='white',
                stopwords=stopwords,
                max_words=100,
                max_font_size=70,
                scale=1,
                random_state=2,
                mask=masking
            ).generate(data)

        else:
            wordcloud = WordCloud(
                background_color='white',
                stopwords=stopwords,
                max_words=100,
                max_font_size=70,
                scale=1,
                random_state=2
        ).generate(data)

        if show == True:
            fig = plt.figure(1, figsize=(5+size, 5+size))
            plt.axis('off')
            if title:
                fig.suptitle(title, fontsize=20)
                fig.subplots_adjust(top=2.3)

            plt.imshow(wordcloud)
            plt.show()

        if makefile == True:
            nn = '{}.jpg'.format(name)
            wordcloud.to_file(nn)

    def most_pop_ngrams(self,n,pop, text_list):
        counts = Counter()
        wds = []

        for text_string in text_list:
            text = text_string.split()
            nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
            filtered = [w for w in text if nonPunct.match(w)]
            textN = self.make_ngram(n,filtered)
            #print(filtered)
            #print(textN[0:10])
            counts.update(textN)
            wds.append(textN)
        return wds

    def most_pop_words(self,n, text_list):
        counts = Counter()

        for text_string in text_list:
            text = text_string.split()
            nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
            filtered = [w for w in text if nonPunct.match(w)]
            counts.update(filtered)
            return counts.most_common(n)

    def update_db(self,bus_id, high_list, low_list,user,pw):
        '''
        Inserts word use into a Database Table

        Args:
            bus_id: a string containing the business ID
            high_list: the list of words from the high set
            low_list; the list of words from the low set
            user: username for the database
            pw: username for the database

        Returns:
            Information is inserted into the yelp_db.b_eval table
        '''

        update_query ='''INSERT INTO yelp_db.b_eval (bus_id, high_list, low_list) VALUES('{}', '{}', '{}')
        ON DUPLICATE KEY UPDATE
        high_list='{}', low_list='{}'
        '''.format(bus_id, high_list, low_list,high_list, low_list)

        # Open database connection, create cursor
        db = pm.connect("localhost",user,pw,"yelp_db" )
        cursor = db.cursor()

        # execute SQL query using execute() method.
        cursor.execute(update_query)
        db.commit()

        # Fetch data
        db.close()



    def find_info(self,id,db_user, db_pw):
        '''
        Puts info from db into pandas DataFrame

        Args:
            id: a string containing the business ID
            db_user: username for the database
            db_pw: username for the database

        Returns:
            A pandas DataFrame with info pulled from the DB
        '''
        col = ['rev_id','bus_id','name','city','state','text', 'stars', 'date', 'useful', 'funny', 'cool']
        #id = '--ab39IjZR_xUf81WyTyHg'
        d1 = self.dat_to_list(self.find_bus(id,db_user, db_pw))
        df = pd.DataFrame(data = d1,columns = col)
        self.businfo.append (df['bus_id'][0])
        self.businfo.append (df['name'][0])
        self.businfo.append (df['city'][0])
        self.businfo.append (df['state'][0])
        return df

    def vec_ctext(self,dfH,dfL):
        '''
        vectorizes text

        Args:
            dfH: DataFrame of high Reviews
            dfL: DataFrame of low Reviews

        Returns:
            updated dataFrames, the vectors, and the vectorizer
        '''
        dftopbot = pd.concat([dfH,dfL],axis = 0)
        list_len = dftopbot.shape[0]

        vectorizer = TfidfVectorizer(stop_words='english',ngram_range =(0,3))
        vectors = vectorizer.fit_transform(dftopbot['cText'])
        words = vectorizer.get_feature_names()

        #dfH['text_vec']= vectorizer.transform(dfH['cText']).toarray()
        #dfL['text_vec']= vectorizer.transform(dfL['cText']).toarray()

        return dfH, dfL, vectorizer, vectors

    def rep_rev(self,dfH, dfL):
        '''
        Find highly representative reviews

        Args:
            dfH: DataFrame of high Reviews
            dfL: DataFrame of low Reviews

        Returns:
            2 highly representative reviews for positive and negative
        '''
        dfH, dfL, vec, vectors = self.vec_ctext(dfH,dfL)

        dfH['y'] = 1
        dfL['y'] = 0

        dfTB = pd.concat([dfH,dfL],axis = 0)

        X = vectors
        y = dfTB['y']
        clf = BernoulliNB()
        clf.fit(X,y)
        rev_list = []
        for i in range(X.shape[0]):
            c = X[i].toarray()
            rev_list.append([(clf.predict_proba(c)[0][1]),clf.predict(c)[0],(dfTB['text'].iloc[i]),len((dfTB['text'].iloc[i]))])

        df_rev = pd.DataFrame(rev_list, columns = ['score','outcome','rev','tlen'])

        best_rev = list(df_rev[(df_rev['outcome']==1)&(df_rev['tlen']> df_rev['tlen'].quantile(q=.45))].sort_values('score',ascending=False).head(2).rev)
        worst_rev = list(df_rev[(df_rev['outcome']==0)&(df_rev['tlen']> df_rev['tlen'].quantile(q=.45))].sort_values('score',ascending=False).head(2).rev)

        return best_rev,worst_rev



    def run(self,id):
        #if ng > 0:
        #    self.ngram = ng
        self.id = id
        cred1 = self.cred
        (self.db_user, self.db_pw) = self.get_cred('cred.csv')

        df = self.find_info(self.id,self.db_user, self.db_pw)
        dfHigh = df[df['stars'] > 3]
        dfLow = df[df['stars'] < 3]
        ctH = self.clean_text(dfHigh['text'])
        ctL = self.clean_text(dfLow['text'])

        numgram = self.ngram

        high_list = self.most_pop_ngrams(self.ngram,100000,ctH)
        low_list = self.most_pop_ngrams(self.ngram,100000,ctL)

        (hl,ll) = self.string_pop(self.ngram,high_list,low_list)

        return(id,hl,ll,self.db_user, self.db_pw)



    def get_name(self):
        return self.businfo

    def run2(self,id):
        #if ng > 0:
        #    self.ngram = ng
        self.id = id
        cred1 = self.cred
        (self.db_user, self.db_pw) = self.get_cred('cred.csv')

        df = self.find_info(self.id,self.db_user, self.db_pw)
        dfHigh = df[df['stars'] > 3]
        dfLow = df[df['stars'] < 3]

        ctH = self.clean_text(dfHigh['text'])
        ctL = self.clean_text(dfLow['text'])

        dfHigh['cText'] = self.clean_text(dfHigh['text'])
        dfLow['cText'] =self.clean_text(dfLow['text'])

        #dfHigh['textList'] = dfHigh['cText'].split()
        #dfLow['textList'] = dfLow['cText'].split()

        #numgram = self.ngram

        #wordlist,tok_sent = self.make_list(ctH)
        #bigram = []
        #for t in tok_sent:
        #    bigram = self.make_ngram(2,t)

        #high_list = self.most_pop_ngrams(self.ngram,5,ctH)
        #low_list = self.most_pop_ngrams(self.ngram,5,ctL)

        #(hl,ll) = self.string_pop(self.ngram,ctH,ctL)

        #a = self.tie_ngram(high_list)
        #b = self.tie_ngram(low_list)
        return(dfHigh,dfLow)
