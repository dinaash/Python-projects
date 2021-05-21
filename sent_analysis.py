import time
import requests
from datetime import datetime
import pickle
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import nltk
#nltk.download("vader_lexicon")
import matplotlib.pyplot as plt
#nltk.download('punkt')

def main():
    print("hi")
    df_2019 = loadData(1554073200,1554678000, "2019_data.pkl")
    print(df_2019.head())
    print(df_2019.tail())
    
    df_2021 = loadData(1617231600,1617836400,"2021_data.pkl")
    print(df_2021.head())
    print(df_2021.tail())

    limit = len(df_2019.days.value_counts().tolist())
    
    x = list(range(1,9))
    fig, ax = plt.subplots()
    ax.plot(x, totalSummation(limit, df_2019), color = "black", label = "before")
    ax.plot(x, totalSummation(limit, df_2021), color = "red", label = "later")
    leg = ax.legend()
    plt.show()


def totalSummation(limit, df):

    limit = len(df.days.value_counts().tolist())
    total_summation = []
    for i in range (1, limit+1):
        oneTotal = df['polar_sent'][df['days']==i].sum()
        total_summation.append(oneTotal/df["days"].value_counts().tolist()[i-1])
    return total_summation 

def loadData(start,end,file_name):
    
    url = "https:://api.pushshift.io/reddit/search"
    list_class = pullshiftpull(start,end)
    picklefile = open(file_name,"wb")
    pickle.dump(list_class, picklefile)
    picklefile = open(file_name,"rb")
    data_class = pickle.load(picklefile)

    dates = []
    text = []
    for cls in data_class:
        dates.append(cls.created_utc)
        text.append(cls.body)
    df = pd.DataFrame({"dates":dates,"body":text})
    df['days']=df.dates.apply(extractDay)
    df['polar_sent']=df.body.apply(determineSentimentPolarity)

    return df
    


def determineSentimentPolarity(row):

    
    row_score = 0
    sid = SentimentIntensityAnalyzer()

    for sentence in tokenize.sent_tokenize(row):
        row_score += sid.polarity_scores(sentence)["compound"]
  
    return row_score
    

def extractDay(row):
    dt = datetime.fromtimestamp(row)
    return dt.day

class redditSubmission:
    def __init__(self):
        self.body = ""
        self.created_utc = ""

def pullshiftpull(start_stamp,end_stamp):
    
    subreddit = "ireland"

    url = "https://api.pushshift.io/reddit/search/?limit=100&after={}&before={}&subreddit={}"
    list_class = []

    while start_stamp < end_stamp:
        
        time.sleep(1) 

        update_url = url.format(start_stamp, end_stamp, subreddit)

        
        json = requests.get(update_url)
        print(json.status_code)

        if json.status_code !=200:
            continue
        else:
            json_data = json.json()
            if "data" not in json_data:
                break
            else:
                json_data = json_data['data']
                if len(json_data)==0:
                    print("no more data")
                    break
                try:
                    start_stamp = json_data[-1]['created_utc']
                except:
                    start_stamp = end_stamp
                list_class = processJsonData(json_data, list_class)
                print(datetime.fromtimestamp(start_stamp))
                print()
        
        
    return list_class
        

def processJsonData(json_data, list_class):
    for item in json_data:
        redCls = redditSubmission()
        redCls.body = item['body']
        redCls.created_utc = item['created_utc']
        list_class.append(redCls)
    return list_class



if __name__== "__main__":
    main()
