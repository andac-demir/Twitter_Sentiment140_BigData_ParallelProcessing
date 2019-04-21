# Opinion Mining and Sentiment Analysis Classification using Twitter Sentiment140 Dataset and PySpark MLlib 

We implement two classiﬁers based on a logistic regression model to decide whether the sentiment in a tweet is positive or negative. The methods applied in this paper constitute of (i) converting the textual data to a numeric form using Term Frequency (TF) hashing to generate feature vectors and Inverse Document Frequency (IDF) to scale the feature vector to reduce the weights of words that appear often in tweets and not affect the sentiment classiﬁcation (ii) tokenizing the textual data into consecutive sequences of words, called n-grams, and then ﬁtting the features (extracted and scaled the same way as described in (i) and then assembled together) from the unigrams, bigrams and trigrams into a logistic regression model. We compare the computational performance and accuracy on validation and test sets for each method.

We worked with Spark’s Dataframe as data structure in this project and used PySpark MLlib’s Pipeline which is an abstraction speciﬁed as a sequence of stages. These stages are run in order and the input DataFrame is transformed as it passes through each stage: tokenizing, extracting feature vectors by HashingTF, and scaling feature vectors by the TFIDF scores. For classsification of the tweet sentiment, we fit the scaled features in a logistic regression model at the end.

## Dataset
Download the Sentiment140 dataset [here](http://help.sentiment140.com/for-students) in a directory named "Data" inside your Workspace.

## Running
First you must parse and clean the dataset with:

```
python preprocessing.py
```

Then extract, scale and fit the fixed size feature vectors in a logistic model with:

```
python train_model.py
```

### Arguments for Training
--ngrams, type=int, default=3, help='n-grams for feature extraction'

--classifier, type=int, default=0, help='use tfidf hashing if 0, otherwise n-grams'

## Results

|   Method   |  Validation Accuracy | Test Accuracy |
| ---------- | -------------------- | ------------- |
|  TFIDF LR  |        0.7883        |     0.7880    |
| BIGRAM LR  |        0.7391        |     0.7398    |
| TRIGRAM LR |        0.6610        |     0.6617    |
|  U+B+T LR  |        0.8079        |     0.8814    |


|   Method   |  Validation ROC-AUC  | Test ROC-AUC  |
| ---------- | -------------------- | ------------- |
|  TFIDF LR  | 0.8576         | 0.8565  |
| BIGRAM LR  | 0.8174         | 0.8174  |
| TRIGRAM LR | 0.7278         | 0.7275  |
|  U+B+T LR  | 0.8821         | 0.8814  |

## References
1. Go, A., Bhayani, R., and Huang, L. Twitter sentiment classiﬁcation using distant supervision. CS224N Project Report, Stanford, 1(12):2009, 2009. 

2. Kim, R. Another twitter sentiment analysis with pythonŁŁpart 5 (tﬁdf vectorizer, model comparison, lexical approach), 2018. 

3. Press, S.J. and Wilson, S. Choosing between logistic regression and discriminant analysis. Journal of the American Statistical Association, 73(364):699–705, 1978.
