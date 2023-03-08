import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from glob import glob
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

import matplotlib.pyplot as plt

def main():

    rawFolderPaths = glob("./database/train/*/")

    categories = []

    for i in rawFolderPaths:
        category = pathfinder(i)
        categories.append(category)

    print(categories)
    print("CatgInicial> " + str(len(categories)))

    categories = ['ciencia', 'comida', 'cotidiano', 'educacao', 'esporte', 'mercado', 'mundo', 'poder', 'saopaulo', 'tec', 'turismo', 'tv']
    print("CatgSelect> " + str(len(categories)))

    # Load the training data
    print ('\nLoading the dataset...\n')
    docs_to_train = datasets.load_files("./database/train/", description=None, categories=categories, load_content=True, 
        encoding='utf-8', shuffle=True, random_state=42)

    print("Target> " + str(docs_to_train.target))

    print("Target> 0 até 10 " + str(docs_to_train.target[:10]))

    for t in docs_to_train.target[:10]:
        print(docs_to_train.target_names[t])

    # Tokenizing text with scikit-learn
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(docs_to_train.data)
    print(X_train_counts.shape)

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(X_train_tf.shape)
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)

    # Training a classifier

    # Some tests
    clf = MultinomialNB().fit(X_train_tfidf, docs_to_train.target)
    docs_new = ['Qual é a explicação científica para que alguns pratos fiquem mais gostosos no dia seguinte',
                'O jogador fez 10 gols, isso é um recorde no campeonato italiano',
                'Os consumidores estão saindo mais as ruas',
                'Veja hoje no Jornal Nacional todas as informações de como renovar a carteira nacional de trânsito']

    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

    print("\n=====================================\n")

    for doc, category in zip(docs_new, predicted): 
        print('%r => %s' % (doc, docs_to_train.target_names[category]))

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf.fit(docs_to_train.data, docs_to_train.target)

    # Evaluation of the performance on the test set
    docs_to_test = datasets.load_files("./database/test/", description=None, categories=categories, load_content=True, 
        encoding='utf-8', shuffle=True, random_state=42)

    docs_test = docs_to_test.data
    predicted = text_clf.predict(docs_test)

    print("\n=====================================\n")
    print("First accuracy evaluation> " +str(np.mean(predicted == docs_to_test.target)))

    text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
    ])

    print(metrics.classification_report(docs_to_test.target, predicted, target_names=docs_to_test.target_names))

    print(metrics.confusion_matrix(docs_to_test.target, predicted))

    print("\n=====================================\n")

    # Do better?
    text_clf.fit(docs_to_train.data, docs_to_train.target)
    predicted = text_clf.predict(docs_test)
    print("Second accuracy evaluation> " + str(np.mean(predicted == docs_to_test.target)))

    print("\n")
    print(metrics.classification_report(docs_to_test.target, predicted, target_names=docs_to_test.target_names))

    print(metrics.confusion_matrix(docs_to_test.target, predicted))

def pathfinder(targetPath):
    path_string = targetPath.replace('./database/train','')
    path_string = path_string.strip('/')
    return path_string

if __name__ == '__main__':
    main()