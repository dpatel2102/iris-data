#Name: Dhruvi Patel
#Id  : 1001833435
import numpy as np
import pandas as pd
import re
import string
from collections import defaultdict
import operator
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

def pre_processing(str_arg):
    preprocessed_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) 
    preprocessed_str=re.sub('(\s+)',' ',preprocessed_str) 
    preprocessed_str=preprocessed_str.lower() 
    return preprocessed_str

class NaiveBayesClassifier:
    def __init__ (self, unique_classes):
        self.classes = unique_classes
    
    def addToBow  (self, data, dict_index):
        if isinstance (data, np.ndarray):
            data = data[0]
     
        for word in data.split(): 
            self.bow_dicts [dict_index][word] += 1
                
    def train (self, dataset, labels):
        self.examples = dataset
        self.labels = labels
        self.bow_dicts = np.array ([defaultdict (lambda:0) for index in range (self.classes.shape[0])])
     
        if not isinstance (self.examples, np.ndarray):
            self.examples = np.array (self.examples)
        
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array (self.labels)
        for index, cat in enumerate (self.classes):
            allcat = self.examples [self.labels == cat] 
            cleandata = [pre_processing (cat_example) for cat_example in allcat]
            cleandata = pd.DataFrame(data = cleandata)
            np.apply_along_axis(self.addToBow, 1, cleandata, index)
        probclass = np.empty (self.classes.shape[0])
        words = []
        catcounts = np.empty (self.classes.shape[0])
        
        for index, cat in enumerate (self.classes):
            probclass [index] = np.sum (self.labels == cat) / float (self.labels.shape[0]) 
            count = list (self.bow_dicts [index].values())
            catcounts [index] = np.sum (np.array (list (self.bow_dicts[index].values()))) + 1 
            words += self.bow_dicts [index].keys()
            self.vocab = np.unique (np.array (words))
            self.vocab_length = self.vocab.shape [0]
            denoms = np.array ([catcounts[index] + self.vocab_length + 1 for index, cat in enumerate (self.classes)])                                                                          
            self.cats_info = [(self.bow_dicts [index], probclass [index], denoms [index]) for index, cat in enumerate (self.classes)]                               
            self.cats_info = np.array (self.cats_info) 

    def ExampleProbability (self, testdata):
        likelihoodprob = np.zeros (self.classes.shape[0])
        for index, cat in enumerate (self.classes): 
            for testdata_token in testdata.split():
                testdata_token_counts = self.cats_info [index][0].get(testdata_token, 0) + 1
                testdata_token_prob = testdata_token_counts / float (self.cats_info [index][2])
                likelihoodprob [index] += np.log (testdata_token_prob)
        post_prob = np.empty (self.classes.shape[0])
        for index, cat in enumerate (self.classes):
            post_prob[index] = likelihoodprob[index] + np.log(self.cats_info[index][1])
        return post_prob
    
    def test(self, testdata):
        prediction = []
        for data in testdata:
            cleandata = pre_processing (data)
            post_prob = self.ExampleProbability (cleandata)
            prediction.append (self.classes[np.argmax (post_prob)])
        return np.array (prediction)

news = skd.load_files ("data", encoding = 'ISO-8859-1')
print ("Successfully Imported Data.")

newslen = len(news.data)
print ("Total size: ",newslen)

newsdata_train, newsdata_test, newstarget_train, newstarget_test = train_test_split (news.data, news.target, test_size = 0.5)
print ("Successfully Splitted data.")
print ("Training data: ",len (newsdata_train))
print ("Testing data: ",len (newsdata_test))

naive_bayes = NaiveBayesClassifier (np.unique (newstarget_train))
print ("Training of Data is in Progress.")

naive_bayes.train (newsdata_train, newstarget_train) 
print ('Training of Data is Completed.')
print ("Targets are Predicted.\n")

predicted_newstarget_test = naive_bayes.test (newsdata_test)

acc = accuracy_score (newstarget_test, predicted_newstarget_test) * 100
print (metrics.classification_report (newstarget_test, predicted_newstarget_test, target_names = news.target_names))
print ("Size of test data set: ",len (newstarget_test))
print ("Accuracy: %.2f" % acc, "%")