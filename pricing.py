import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import string
import glob
import re
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.feature_selection import SelectKBest, f_regression # pearson: numerical input and output
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from google.colab import files


class Preprocessing:

  """ 
  This class reads and preprocesses data

  Arguments:
    train_add: The address of training set
    test_add: The address of test set
    stop_words_add: The address of stopwords
  """

  def __init__(self,train_add,test_add,stop_words_add):
    self.train_add = train_add
    self.test_add = test_add
    self.stop_words_add = stop_words_add

  def read_data(self,idn):
    if idn == 'train':
      dataset = pd.read_csv(self.train_add)
      price = dataset['price'].tolist()
    else:
      dataset = pd.read_csv(self.test_add)
    id = dataset['id'].tolist()
    product_description = dataset['product_description'].tolist()
    if idn == 'train':
      data = [id,product_description,price]
    else:
      data = [id,product_description]
    
    return data
  
  def remove_nonalphabetic(self, desc):
    new_product_des = []
    for i in range(len(desc)): 
      des = desc[i]
      des_dict = ast.literal_eval(des)
      new_des = {}
      for key, value in des_dict.items():
        k = str(key).replace('\u200c', ' ')
        k = k.replace('\\r', ' ')
        k = k.replace('\r\n', ',')
        k = k.replace('\\', ' ')
        k = k.replace('\\n', ' ')
        v = str(value).replace('\u200c', ' ')
        v = v.replace('u200c', ' ')
        v = v.replace('\\r', ' ')
        v = v.replace('\r\n', ',')
        v = v.replace('\\', ' ')
        v = v.replace('\\n', ' ')
        v = v.replace('[', '')
        v = v.replace(']', '')
        v = v.replace("'", '')
        v = v.replace("/", '')
        new_des[k] = v
      new_product_des.append(new_des)
    return new_product_des 

  def split_brand_cat(self,des_list):
    categories = []
    brands = []
    descriptions = []
    for i in range(len(des_list)):
      brands.append(des_list[i]['برند'])
      categories.append(des_list[i]['دسته بندی'])
      del des_list[i]['برند']
      del des_list[i]['دسته بندی']
      des = ''
      for key,value in des_list[i].items(): 
        des = des + ',' + key + ' ' + value
      descriptions.append(des)
    new_des = [categories,brands,descriptions]
    return new_des

  def remove_arabic_char(self,word):
    chars = ['ْ','ٌ','ٍ','ً','ُ','ِ','َ','ّ','إ','لأ','آ','لآ','ة','ؤ','ء','ئ','ك','ي']
    new_chars = ['','','','','','','','','ا','لا','ا','لا','ه','و','','ی','ک','ی']
    new_word = word
    for i in word:
      if i in chars:
        new_word = new_word.replace(i,new_chars[chars.index(i)])
    return new_word
  
  def remove_punc_str(self,word):
    punctuation = list(string.punctuation) + ['؛','٬','؟','«','»','ـ','،','×','٪','﷼','…','[',']']
    float_num = re.findall('\d+\.\d+',word)
    new = word
    if float_num :
      for i in float_num :
        new = new.replace(i,str(int(float(i))))
    new_word = []
    for letter in list(new):
      if letter in punctuation :
        new_word.append(' ')
      else: 
        new_word.append(letter)
    new_word_str = ''.join([str(elem) for elem in new_word])
    return re.sub(' +', ' ', new_word_str)
  
  def persian_stopwords(self):
    file = open(self.stop_words_add, 'rt')
    text = file.readlines()
    persian_stop_words = [x.strip() for x in text]
    file.close()
    return persian_stop_words

  def preprocess(self,idn):
    preprocessed_des = []
    data = self.read_data(idn)
    new_des = self.remove_nonalphabetic(data[1]) 
    # extract brand name and category from descriptions
    new_des = self.split_brand_cat(new_des)
    stop_words = self.persian_stopwords()
    for i in range(len(new_des[2])):
      col = new_des[2][i]
      new_col = self.remove_punc_str(col)
      new_col = self.remove_arabic_char(new_col)
      col_list = new_col.split(' ')
      new_list = []
      for word in col_list:
        if word not in stop_words:
          new_list.append(word)
      final_col = [ele for ele in new_list if ele != '']
      if final_col == []:
        preprocessed_des.append(['missing'])
      else :
        preprocessed_des.append(final_col)
    
    if idn =='train':
      new_data = [data[0],new_des[0],new_des[1],preprocessed_des,data[2]] # id, category, brand, preprocessed description, price
    else:
      new_data = [data[0],new_des[0],new_des[1],preprocessed_des] # id, category, brand, preprocessed description

    return new_data

  def remove_unknown_brands(self,brands, brands_test):
    """
    This function removes some brands from the test set because they are in the test set but not in the training set.
    If we don't remove unknown brands, we'll face a problem in the feature extraction phase.
    """
    a = 'متفرقه'
    for i in brands_test:
      if i not in brands:
        idx = brands_test.index(i)
        brands_test[idx]= a 
    return brands_test
    
    
class FeatureExtraction:
  """
  Extract features from different columns according to their type
  """
  def onehot_encoder(self,col_train, col_test):
    """
    creates onehot vector for categorical data
    """
    values_train = col_train.values
    values_test = col_test.values
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values_train)
    integer_encoded_test = label_encoder.transform(values_test)

    # binary encode
    one_hot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)
    onehot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    onehot_encoded_test = one_hot_encoder.transform(integer_encoded_test)

    return onehot_encoded,onehot_encoded_test
  
  def prepare_data_for_vectorizers(self, col):
    """
    creates a sequence of words which are seperated by space
    """
    input = []
    list_of_words = col.values.tolist()
    for i in range(len(list_of_words)):
      new = ' '.join([word for word in list_of_words[i]])
      input.append(new)
    return input
  
  def bag_of_words(self,col,col_test,ngram, min_df, max_features):
    """
    converts text features to numbers using bag of words vectrizer
    """
    col = self.prepare_data_for_vectorizers(col)
    col_test = self.prepare_data_for_vectorizers(col_test) 
    vectorizer = CountVectorizer(ngram_range=(1, ngram), min_df=min_df, max_features=max_features)
    bow = vectorizer.fit_transform(col)
    bow_test = vectorizer.transform(col_test)
    return bow, bow_test

  def tfift_vectorizer(self,col,col_test,ngram, min_df, max_features):
    """
    converts text features to numbers using tfidf vectrizer
    """
    col = self.prepare_data_for_vectorizers(col)
    col_test = self.prepare_data_for_vectorizers(col_test) 
    vectorizer = TfidfVectorizer(ngram_range=(1, ngram), min_df=min_df, max_features=max_features)
    tfidf = vectorizer.fit_transform(col)
    tfidf_test = vectorizer.transform(col_test)
    return tfidf,tfidf_test

  def consolidate_features(self,features):
    all_features = hstack((features[0], features[1],features[2])).tocsr()
    return all_features
    
    
class MakePrediction:
  """
  This class contains the two most successful machine learning algorithms in this task
  """
  def __init__(self, idn, model_address, test_id):
    self.idn = idn
    self.address = model_address
    self.test_id = test_id
  
  def feature_selection(self,X_train,y_train,X_val,X_test,num_of_features):
    """
    Feature selection using pearson correlation coefficient
    """
    fselect = SelectKBest(score_func=f_regression, k=num_of_features)
    train_features = fselect.fit_transform(X_train, y_train)
    val_features = fselect.transform(X_val)
    test_features= fselect.transform(X_test)
    return train_features,val_features,test_features

  def ml(self,train_features, train_labels, test_features, split_ratio, selection = True, num_of_features = 500):
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=split_ratio)

    if selection == True:
      X_train,X_val, X_test = self.feature_selection(X_train,y_train,X_val,test_features,num_of_features)
    else:
      X_test = test_features

    if self.idn == 'LR':
      regressor = LinearRegression()
      regressor.fit(X_train, y_train)
      preds = regressor.predict(X_val)
      print('MAPE on validation set: ',mape(preds,y_val))
      # save model
      pickle.dump(regressor, open(self.address, 'wb'))

    elif self.idn == 'RF':
      estimators = list(range(1,30))
      depths = [2,3,4,5,6,7]
      results = []
      for i in estimators:
        for j in depths :
          regressor = RandomForestRegressor(n_estimators=i+1, max_depth=j)
          regressor.fit(X_train, y_train) 
          preds = regressor.predict(X_val)
          final_score = mape(preds,y_val)
          filename = self.address + '_e' + str(i) + '_d' + str(j) +'_'+ str(final_score)+'.sav'
          pickle.dump(regressor, open(filename, 'wb'))
          results.append(final_score)
      best_addresses = glob.glob(self.address + '_e*_d*_' + str(min(results)) + '.sav')
      print('The address of best RF: ', best_addresses[0])
      print('Best RF is going to be downloaded!')
      files.download(best_addresses[0])
      regressor = pickle.load(open(best_addresses[0], 'rb'))
    test_labels = regressor.predict(X_test)
    test_labels_list = test_labels.tolist()

    # remove negative values because price is always positive
    for i in range(len(test_labels_list)):
      if test_labels_list[i] < 0:
        test_labels_list[i] = 0
    
    result = pd.DataFrame(list(zip(self.test_id, test_labels_list)),columns=['id', 'price'])
    return result
    
class Pricing:

  def __init__(self,train_add,test_add, stopwords_add, vectorizer, model, model_add, split_ratio, selection, num_features):
    self.train_add = train_add
    self.test_add = test_add
    self.stopwords_add = stopwords_add
    self.vectorizer = vectorizer
    self.model = model
    self.model_add = model_add
    self.split_ratio = split_ratio
    self.selection = selection
    self.num_features = num_features

  def main(self):
    # Preprocessing
    print("-----Preprocessing-----")
    preprocess = Preprocessing(self.train_add,self.test_add,self.stopwords_add)
    new_training_data = preprocess.preprocess('train')
    new_test_data = preprocess.preprocess('test')
    brands_train = new_training_data[2]
    brands_test = new_test_data[2]
    new_brands_test = preprocess.remove_unknown_brands(brands_train,brands_test)
    new_test_data[2] = new_brands_test
    new_data = pd.DataFrame(list(zip(new_training_data[0],new_training_data[1],new_training_data[2],new_training_data[3],new_training_data[4])),
               columns =['id', 'category','brand','description','price'])
    new_data_test =  pd.DataFrame(list(zip(new_test_data[0],new_test_data[1], new_test_data[2],new_test_data[3])),
                  columns =['id', 'category','brand','description'])
    
    # Feature extraction
    print("-----Feature extraction-----")
    fe = FeatureExtraction()
    onehot_brands, onehot_brands_test = fe.onehot_encoder(new_data['brand'],new_data_test['brand'])
    onehot_categories, onehot_categories_test = fe.onehot_encoder(new_data['category'],new_data_test['category'])
    if vectorizer == self.vectorizer:
      des_vector, des_test_vector = fe.tfift_vectorizer(new_data['description'],new_data_test['description'],3,3,500000)
    else:
      des_vector, des_test_vector = fe.bag_of_words(new_data['description'],new_data_test['description'],3,3,500000)
    features = fe.consolidate_features([onehot_categories,onehot_brands,des_vector])
    features_test = fe.consolidate_features([onehot_categories_test,onehot_brands_test,des_test_vector])

    # Prediction
    print("-----Prediction-----")
    predictor = MakePrediction(self.model, self.model_add, new_data_test['id'])
    result = predictor.ml(features,new_data['price'],features_test,self.split_ratio,self.selection,self.num_features)
    return result
    
    
train_add = '/content/data/train.csv'
test_add = '/content/data/test.csv'
stopwords = '/content/Persian_stopwords.txt'
model_add = '/content/LR.sav'
pricing = Pricing(train_add,test_add,stopwords,'bow','LR',model_add,0.33,True,200)
result = pricing.main()
result.to_csv('/content/output.csv', index=False)
