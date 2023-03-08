import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
import os

def main():

    series = pd.read_csv('./articles.csv', header=0, parse_dates=True, squeeze=True, usecols=[0,1,3])

    # categorias disponíveis
    print(series['category'].unique())

    categories = series['category'].unique()

    dataUnified = 3

    if dataUnified == 0:
        for c in categories:
            os.mkdir("./database/" + c + "-test")
            os.mkdir("./database/" + c + "-train")

        ind = 10000
        for cat in categories:

            tempData = series[series['category'].str.contains(cat)]
            toFile = pd.DataFrame({'category' : tempData['category'].astype(str) ,'news' : tempData['title'].astype(str) + ". " + tempData['text'].astype(str)})
            
            for data in toFile.values:
                if ind % 4 == 0:
                    fileSave = open("./database/" + cat + "-test/" + str(ind), "w")
                else:
                    fileSave = open("./database/" + cat + "-train/" + str(ind), "w")

                fileSave.writelines(data[1]) # only text, no category
                fileSave.close()
                ind += 1

    elif dataUnified == 0:
        for c in categories:
            os.mkdir("./database/" + c)

        ind = 10000
        for cat in categories:

            tempData = series[series['category'].str.contains(cat)]
            toFile = pd.DataFrame({'category' : tempData['category'].astype(str) ,'news' : tempData['title'].astype(str) + ". " + tempData['text'].astype(str)})
            
            for data in toFile.values:
                fileSave = open("./database/" + cat + "/" + str(ind), "w")
                fileSave.writelines(data[1]) # only text, no category
                fileSave.close()
                ind += 1
    else:
        for c in categories:
            os.makedirs("./database/test/" + c, exist_ok=True)
            os.makedirs("./database/train/" + c, exist_ok=True)

        ind = 10000
        for cat in categories:

            tempData = series[series['category'].str.contains(cat)]
            toFile = pd.DataFrame({'category' : tempData['category'].astype(str) ,'news' : tempData['title'].astype(str) + ". " + tempData['text'].astype(str)})
            
            for data in toFile.values:
                if ind % 4 == 0:
                    fileSave = open("./database/test/" + cat + "/" + str(ind), "w")
                else:
                    fileSave = open("./database/train/" + cat + "/" + str(ind), "w")

                fileSave.writelines(data[1]) # only text, no category
                fileSave.close()
                ind += 1

if __name__ == '__main__':
    main()