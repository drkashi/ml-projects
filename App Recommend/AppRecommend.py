# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
import os
import argparse

# Modules for Http
import requests
import json
# package for flattening json in pandas df
from pandas.io.json import json_normalize

# NLP modules for manipulating text data in app (nltk.download)
import re
import string
import nltk

# Sklearn for all modeling tasks
import sklearn.preprocessing as preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', 2000)

''' 
###################################
HELPER FUNCTIONS, CLASSES and UTILS 
###################################
'''


'''
Grouping Features by their types for further transformations
'''

TXT_tags = ['description', 'advisories', 'genres', 'genreIds', 'languageCodesISO2A']

CAT_tags = ['artistId', 'artistName', 'primaryGenreId', 'primaryGenreName',
            'contentAdvisoryRating', 'trackContentRating', 'currency', 'kind']

NUM_tags = ['averageUserRating', 'averageUserRatingForCurrentVersion']
            
AVG_tags = ['userRatingCount', 'userRatingCountForCurrentVersion'] # use for better ranking based on user rating

USE_tags = TXT_tags + CAT_tags + NUM_tags + ['trackId']

ALL_tags = ['description', 'genres', 'advisories',
            'isGameCenterEnabled', 'kind', 'features',
            'supportedDevices', 'averageUserRatingForCurrentVersion', 'trackCensoredName', 'languageCodesISO2A',
            'fileSizeBytes', 'contentAdvisoryRating', 'userRatingCountForCurrentVersion', 'trackViewUrl',
            'trackContentRating', 'currentVersionReleaseDate', 'trackId', 'trackName', 'currency', 'wrapperType',
            'version', 'artistId', 'artistName', 'price', 'bundleId', 'formattedPrice',
            'releaseNotes', 'isVppDeviceBasedLicensingEnabled', 'primaryGenreName', 'primaryGenreId',
            'sellerName', 'releaseDate', 'minimumOsVersion', 'genreIds', 'averageUserRating', 'userRatingCount']


def AppRecommend(currentAppID, eligibleApps=[]):
    
    # check input app list validity
    appList = AppCreateCombinedList(currentAppID, eligibleApps)
    
    # Create http request from combined list
    app_request = AppCreateRequestStr(appList)
    
    # fetch all apps data into json format
    response = AppRequestRemoteData(app_request)
   
    # parse json response info DataFrame
    dfResponse = LoadResponseToDataframe(response)
    trainData = dfResponse[USE_tags].copy()
    trainData.set_index('trackId', inplace=True)

    # check if currentAppID returned with data
    try:
        trainData.loc[currentAppID]
    except:
        print("ERROR: invalid/missing App ID:", currentAppID)
        return

    # get featured dataset in a new Dataframe
    print("INFO: perform feature engineering...")
    X_features = AppFeatureEngineering(trainData)
    
    # compute pairwise cosine similarity (among other metrics)
    score_df = get_cosine_similarity(X_features, currentAppID)
    
    # save ranking into file
    df = save_recommendation_file(score_df, currentAppID)


'''
Utils for manipulating json files request and response
'''


# create combined list of current app and eligble apps
def AppCreateCombinedList(currentAppID, eligibleApps):
    try:
        appList = eligibleApps
        len(eligibleApps)
    except TypeError:
        appList = [eligibleApps]
    appList.append(currentAppID)
    return list(set(appList))


def AppCreateRequestStr(appList):
    app_str = ','.join([str(x) for x in appList])
    app_request = 'http://itunes.apple.com/lookup?id=' + app_str
    return app_request
    

# request and download all apps for the first time
def AppRequestRemoteData(request):
    try:
        response = json.loads(requests.get(request).text)
    except:
        print("ERROR: occurred while fetching remote App data!")
    return response

    
# create http request, get the app data, save into file and return json string
def RequestAndSaveAppFile(appList):
    try:
        len(appList)
    except TypeError:
        appList = [appList]
    for appId in appList:
        appRequest = 'http://itunes.apple.com/lookup?id=' + str(appId)
        saveToFile = mainDir + '/app_' + str(appId) + '.txt'
        data = json.loads(requests.get(appRequest).text)
        with open(saveToFile, "w") as fileWriter:
            json.dump(data, fileWriter)
    return data


# load json file for single app
def LoadAppInfoFromFile(fileName):
    with open(fileName, 'r') as fileReader:
        data = json.load(fileReader)
    return data


# get json string for aingle or multiple apps and create DataFrame
def LoadResponseToDataframe(response):
    data = response['results'][0]
    origDf = json_normalize(response['results'])
    return origDf


'''
Customized Tokenize/Cleaning Functions
'''

# Functions to remove punctuation, tokenize, remove stopwords and Stemming/Lemmarizing
stopwords = nltk.corpus.stopwords.words('english')


# cleaning, tokenizing and stemming text features
def CleanAndStemmText(text, ngram=1):
    ps = nltk.PorterStemmer()
    text = re.sub(',/', ' ', text)
    text = "".join([word.lower() for word in text if word not in string.punctuation + string.digits])
    text = str(text).replace(',', ' ').replace('/', ' ')
    tokens = re.split('\W', text)
    if ngram > 1:
        text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    else:
        text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


# cleaning, tokenizing and lemmatizing text features
def CleanAndLemmText(text, ngram=1):
    wn = nltk.WordNetLemmatizer()
    text = str(text).replace(',', ' ').replace('/', ' ')
    text = "".join([word.lower() for word in text if word not in string.punctuation + string.digits])
    tokens = re.split('\W', text)
    if ngram > 1:
        text = " ".join([wn.lemmatize(word) for word in tokens if word not in stopwords])
    else:
        text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text


# splitting, tokenizing and lemmatizing text features
def TokenizeAndLemmatizeText(text):
    wn = nltk.WordNetLemmatizer()
    if type(text) is list:
        text = " ".join(text) # convert array to str
    text = str(text).replace(',', ' ').replace('/', ' ')
    text = " ".join(re.findall("[\w]{2,}", text))
    text = "".join([word.lower() for word in text if word not in string.punctuation + string.digits])
    tokens = nltk.word_tokenize(text)
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text


'''
Main part that doing the feature engineering and compose the data set for modeling :
Perform feature Union from the different transformations and preserve feature names. 
'''

'''
Helper Classes for FeatureUnion with DataFrame 
'''

# Modules for interaction between scipy and pandas DataFrame and FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse


# Apply function on column(s) (used to return column from DataFrame
class PandasTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        return self.fn(X)


# Make sure the returned Object is either sparse matrix or Dataframe
class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X, y, **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs


'''
Feature Engineering and Data Vectorization
'''


def AppFeatureEngineering(trainData):

    """
    Notes (for improvement etc.):
    1. Dimensionality reduction on text attributes
    2. Combination of metrics on different feature types, e.g. manhatthan dist for categorical/One-Hot
    3. Different Wieights for different types of features
    4. Clustering of wider/entire space
    5. Caching the results for future fast online retrieval
    """
    # define tokenizer for text attributes - TEX_tags
    spec_tokenizer = lambda x: TokenizeAndLemmatizeText(x)
    
    # In this case, TEXT features can be an array of arrays of strings per document and not just strings.
    # In order to be able to run Fit, we exclude the prepocessor step (because there are no raw documents):
    
    tfidf_vect = TfidfVectorizer(input='content', preprocessor=lambda x: x, tokenizer=spec_tokenizer,
                                 analyzer='word', stop_words='english', ngram_range=(1,2))
    
    # tfidf for list/arrays that do not need special tokenizing, e.g. 'genreIds'
    tfidf_list = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1,1))
    
    # build feature union pipileline to transform and combine the different feature types
    union = PandasFeatureUnion(
        transformer_list = [
            ('description', Pipeline([
                ('select', PandasTransformer(lambda x: x['description'])),
                ('tfidf',  tfidf_vect),
                ('dframe', PandasTransformer(lambda x: pd.DataFrame(x.toarray(),columns=tfidf_vect.get_feature_names())))
            ])),

            ('advisories', Pipeline([
                ('select', PandasTransformer(lambda x: x['advisories'])),
                ('tfidf',  tfidf_vect),
                ('dframe', PandasTransformer(lambda x: pd.DataFrame(x.toarray(),columns=tfidf_vect.get_feature_names())))
            ])),

            ('genreIds',   Pipeline([
                ('select', PandasTransformer(lambda x: x['genreIds'])),
                ('tfidf',  tfidf_list),
                ('dframe', PandasTransformer(lambda x: pd.DataFrame(x.toarray(),columns=tfidf_list.get_feature_names())))
            ])),

            ('dummies', Pipeline([
                ('select', PandasTransformer(lambda x: pd.get_dummies(x, columns=CAT_tags, drop_first=False))),
                ('index', PandasTransformer(lambda x: x.reset_index()))
            ]))
        ])
    X_union = union.fit_transform(trainData)

    X_union = X_union.drop(axis=1, columns=TXT_tags)

    print("Features shape:", X_union.shape)
    print(type(X_union))

    scaler_minmax = preprocessing.MinMaxScaler(copy=False)
    X_union[NUM_tags] = scaler_minmax.fit_transform(X_union[NUM_tags].values)
    
    X_union.set_index('trackId', inplace=True)
    
    return X_union


'''
Pairwise Metrics Computation
'''


# cosine metric better for orientation similarity
def get_cosine_similarity(X_features, app_id):
    print("INFO: compute Cosine Similarity for AppId: %d" % app_id)
    sc_cosine = cosine_similarity(X_features)
    df_cosine = pd.DataFrame(sc_cosine, index=X_features.index, columns=X_features.index)
    df_cosine = df_cosine.loc[app_id].sort_values(ascending=False)
    
    return df_cosine

# save scoring into csv file
def save_recommendation_file(score_list, app_id):
    file_name = str(app_id) + '_recommend.csv'
    df = pd.DataFrame(score_list)
    df.rename_axis('AppID', inplace=True)
    df.set_axis(['Score'], axis=1, inplace=True)
    df = df.drop([app_id])
    try:
        df.to_csv(file_name)
        print("INFO: scoring saved into: %s" %file_name)
    except:
        print("ERROR: failed saving file: %s" %file_name)
    return df


'''
==> Main
'''
def main():
    current_dir = os.getcwd()
    print("INFO: Working directory: %s" % current_dir)

    # Example: app_list = [429775439, 506627515, 504720040, 447553564, 440045374, 512939461]

    parser = argparse.ArgumentParser(
        description = "Script to create scoring csv file given AppId and eligible App Id's")
    # Arguments
    parser.add_argument(
        '-a', '--app', type=int, help='App Id', required=True)
    parser.add_argument(
        '-e', '--eligible', type=int, help='Eligible comma separated App list', required=True, nargs='+')

    args = parser.parse_args()
    curr_app = args.app
    eligible_apps = args.eligible

    print("\tCurrent AppID: ", curr_app)
    print("\tEligible Apps: ", eligible_apps)

    AppRecommend(curr_app, eligible_apps)


if __name__ == "__main__":
    main()
