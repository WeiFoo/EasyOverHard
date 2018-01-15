## Note:

This is a better implementation of our FSE'17 paper. There are some differences to the paper in FSE'2017:

* We observed that there're some variants in the word2vec model
 due to the randomness, therefore, we'd better verify the stability of 
 our tuning technique.
 
* Here, we report the average results across 10 different word2vec models 
generated with 10 different seeds and report the median values.


# Introduction

This repo is the source code of  [Easy over Hard: A Case Study on Deep learning](https://arxiv.org/abs/1703.00133),
 which is to replicate and comment on the results of
 Bowen Xu et al. [Predicting Semantically Linkable Knowledge in Developer
  Online Forums
 via Convolutional Neural Network.](http://dl.acm.org/citation.cfm?id=2970357) ASE'16.
 
The main idea is, by using some simple technique, like parameter tuning by 
DE, the traditional machine learning method, like SVM, could 
compete with deep learning method for some SE tasks in terms of 
performance scores, while with quite few computing resources and time cost
(10 minutes vs. 14 hours in this case). We offer these results as a cautionary 
tale to the software analytics community and suggest that not every new innovation
 should be applied without critical analysis.
  If researchers deploy some new and expensive process, that work should be 
  baselined against some simpler and faster alternatives.
  
# Reference

```
@inproceedings{fu2017easy,
 author = {Fu, Wei and Menzies, Tim},
 title = {Easy over Hard: A Case Study on Deep Learning},
 booktitle = {Proceedings of the 2017 11th Joint Meeting on Foundations of Software Engineering},
 series = {ESEC/FSE 2017},
 year = {2017},
 location = {Paderborn, Germany},
 pages = {49--60},
 numpages = {12},
 publisher = {ACM},
} 
```



# Setup

__NOTE: only tested on Linux(ubuntu 16.04) and Mac(10.12), sorry for Windows users.__

If you don't want to mess up your local pacakges, please install dependencies
according to ```requirement.txt``` manually.

Or simply run the following command in your terminal

```
sudo pip install -r requirements.txt

```

# Data
The original data is stored in ASEDataset folder. 
The ASE'16 paper's author [Bowen Xu](https://github.com/XBWer/ASEDataset), is
credited with sharing their original training and testing data, thanks! Sharing is great!

```
├── ASEDataset
│   ├── _4_SentenceData.txt
│   ├── testPair.txt
│   └── trainingPair.txt

```

According to Xu et al, mappings between type labels and ID are listed as follows:

``` 
scores in testPair.txt    ID	    type
    1.00              |    1     |  duplicate
    0.8               |    2     |  direct link
    0<x<0.8           |    3     |  indirect link
    0.00              |    4     |  isolated
```


The structure of the training/testing data is stored in pandas.dataframe 
format.


```
>>> train_pd.head
ID,    'PostId', 'RelatedPostId', 'LinkTypeId', 'PostIdVec', 'RelatedPostIdVec', 'Output'
0       283           297          1              [...]         [...]              [...]
1        56            68          2              [...]         [...]              [...]
2         5            16          3              [...]         [...]              [...] 
3      9083          6841          4              [...]         [...]              [...]   
4      5363          5370          1              [...]         [...]              [...]  
5       928           949          2              [...]         [...]              [...]  
...
```
'PostIdVec' and 'RelatedPostIdVec' represent the word embeddings of corresponding posts.

The relationship between 'PostId' (or 'Related PostId') and Sentence in 
'_4_SentenceData.txt' is :

The Nth line of _4_SentenceData.txt is the text of the question whose Id is 
N+1. For example,

```
PostId: 283, refers to the 283th line of text in _4_SentenceData.txt 
RelatedPostId: 297, refers to the 296th line of text in _4_SentenceData.txt 
LinkTypeId:1, menas that they're duplicate.
```




# Run

## 10 word2vec models
* ```pytohn experiment.py```

*  Note that, for each word2vec model, we did a 10-fold cross evaluation for parameter tuning, which also means we evaluate the testing data 10 times using each model. 




# Result 

## Results in FSE paper

### Deault SVM+wordEmbedding


```
                       precision         recall       f1-score        support

       Duplicate    0.724(0.611)   0.525(0.725)   0.609(0.663)            400
          Direct     0.514(0.56)   0.492(0.433)   0.503(0.488)            400
        Indirect    0.779(0.787)    0.980(0.98)   0.864(0.873)            400
        Isolated    0.601(0.676)   0.645(0.538)     0.622(0.6)            400

     avg / total    0.655(0.658)   0.658(0.669)   0.650(0.656)           1600

```


### Tuning SVM+wordEmbedding



```

                       precision         recall       f1-score        support

       Duplicate    0.885(0.898)   0.860(0.898)   0.878(0.898)            400
          Direct    0.851(0.758)   0.903(0.903)   0.841(0.824)            400
        Indirect     0.944(0.84)   0.995(0.773)   0.969(0.805)            400
        Isolated     0.903(0.89)   0.905(0.793)   0.909(0.849)            400

     avg / total    0.896(0.847)   0.897(0.842)   0.899(0.841)           1600


```

## Results of this implementation

Note: The following default SVM results are averaged over 10, while tuning 
SVMs are over 100(10 models X 10-cross eval).

### Deault SVM+wordEmbedding
* Note that the number in parentheses represent Xu's default SVM+wordEmbedding 
method.

```
                       precision         recall       f1-score        support

       Duplicate    0.734(0.611)   0.538(0.725)   0.621(0.663)            400
          Direct     0.516(0.56)   0.497(0.433)   0.508(0.488)            400
        Indirect    0.790(0.787)    0.970(0.98)   0.871(0.873)            400
        Isolated    0.603(0.676)   0.643(0.538)     0.621(0.6)            400

     avg / total    0.660(0.658)   0.662(0.669)   0.655(0.656)           1600

Done!
     
------------------------------------------------------------------------
# Runtime: 65.4 secs

```

### Tuning SVM+wordEmbedding

* Note that the number in parentheses represent Xu's CNN+wordEmbedding method.
* Depending on your hardware, the actual runtime may vary from 9 minutes
 to 20 minutes(our observation) for each fold, totally 90 minutes to 200 
 minutes for one word2vec model experiment.



```

                       precision         recall       f1-score        support

       Duplicate    0.913(0.898)   0.932(0.898)   0.919(0.898)            400
          Direct    0.919(0.758)   0.907(0.903)   0.917(0.824)            400
        Indirect     0.968(0.84)   0.993(0.773)   0.980(0.805)            400
        Isolated     0.944(0.89)   0.927(0.793)   0.935(0.849)            400

     avg / total    0.936(0.847)   0.940(0.842)   0.938(0.841)           1600

Done!

------------------------------------------------------------------------
# Runtime: 6122.8 secs
```

