import creds
import praw
import pprint as pp
import pandas as pd
from praw.models import MoreComments
from operator import attrgetter
import pickle
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

reddit = praw.Reddit(client_id = creds.client_id,\
                     client_secret = creds.client_secret,\
                     user_agent = creds.user_agent, \
                     username = creds.username, \
                     password = creds.password)

class commentClass:
    def __init__(self):
        self._title = ""
        self._comments = []
        self._created = 0
        self._score = 0
        self._id = 0

    def __len__(self):
        return len(self._comments)

    def __getitem__(self, position):
        return self._comments[position]

class masterClass:
    def __init__(self):
        self._class_list = []
        self._index = 0

    def __len__(self):
        return len(self._class_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._class_list):
            raise StopIteration
        index = self._index
        self._index += 1
        return self._class_list[index]

    def sort_length(self, attri_key):
        sorted_comments = sorted(self._class_list, key = attrgetter(attri_key))
        return sorted_comments
    
def main():

    # Load the first 100 hot submissions from two different subreddits
    firstClass = createClass("coronavirus",100)
    secondClass = createClass("luxembourg",100)

    #Create a list of the loaded submissions' titles
    raw_data = []
    #Fill the list with coronavirus submission titles, then luxembourg
    for cls in firstClass:
        raw_data.append(cls._title)
    for cls in secondClass:
        raw_data.append(cls._title)

    #Create labels for each of the titles
    #As the first 100 titles in the list are from the coronavirus subreddit
        #the first 100 lables in the labels list are coronavirus
    labels = ["coronavirus"]*100 + ["luxembourg"]*100
    #Process the loaded titles using the pre-defined function
    data = [preprocess_text(t) for t in raw_data]
    #Create the processing stages that specify the type of vector representation
    #and the desired classifier

    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
    #List parameters for the text vectorizing methods and classifier method
    tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
    }
    
    #Split the data into train set and test set
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.33,
        random_state=42)
    #Fit the multinomial Bayesian classifier with different parameters and 
    #print the model accuracy score with best performing combination of parameter
    clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring='f1')
    clf.fit(x_train, y_train)
    print(classification_report(y_test, clf.predict(x_test), digits=4))

    

def preprocess_text (text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('@[^\s]+','USER', text)
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +',' ', text)
    return text.strip() 

       
def createClass(name, sample_size):
    sub_ids = lookAtSubreddit(name,sample_size)
    maClass = masterClass()
    for id in list(sub_ids)[:sample_size]:
        cls = commentClass()
        cls._title = id.title
        cls._comments = harvestCommentReplies(id)
        cls._created = id.created_utc
        cls._score = id.score
        cls._id = id.id
        maClass._class_list.append(cls)
    return maClass


    

def harvestCommentReplies(id):
    
    sub = reddit.submission(id = id)
    sub.comments.replace_more(limit = None)
    comment_queue = sub.comments[:]
    all_comments = []

    while comment_queue:
        comment = comment_queue.pop(0)
        all_comments.append(comment.body)
        comment_queue.extend(comment.replies)

       
    return all_comments


def lookAtSubreddit(name,sample_size):
    sub = reddit.subreddit(name).hot(limit = sample_size)
    return sub

    

if __name__ == "__main__":
    main()
    
   









    




















               
