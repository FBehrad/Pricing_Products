# Price prediction

Pricing products based on their descriptions

## Steps

In this project, we predict the price of products according to the descriptions provided by sellers. 
The whole process can be divided into four steps:
1) Preprocessing: Removing nonalphabetic characters, punctuations, and Persian stopwords. Replacing Arabic letters with Persian ones.
2) Feature extraction: Extracting brands and categories from descriptions and converting them to one-hot vectors. Using both tfidf and bag of word vectorizers to encode text features. 
3) Feature selection: Choosing a subset of features by Pearson's correlation coefficient. 
4) Price prediction: Predicting the final price using random forest and linear regression.


## Persian Stop Words

Persian_stopwords.txt includes Persian stopwords (this file is collected from all Persian stopwords available on the Internet. However, this version is reviewed and edited.)
