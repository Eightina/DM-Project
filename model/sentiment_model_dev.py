# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:35:51 2023

@author: Neal
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import joblib

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    
    grid_search = GridSearchCV(pipeline, parameters, scoring='f1', n_jobs=-1, verbose=1)
    # df = pd.read_excel('./data/Task-2/train.xlsx')
    df = pd.read_excel('/app/model/data/Task-2/train.xlsx')
    X= df.text
    grid_search.fit(df.text,df.label)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    joblib.dump(grid_search, "text_sentiment_model_v001.joblib")