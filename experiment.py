from __future__ import division, print_function
import pickle
import pdb
import os
import time
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn import metrics
import gensim
import random
from learners import SK_SVM
from tuner import DE_Tune_ML
from model import PaperData
from utility import study
from results import results_process
import numpy as np
import wget
import zipfile


def tune_learner(learner, train_X, train_Y, tune_X, tune_Y, goal,
                 target_class=None):
  """
  :param learner:
  :param train_X:
  :param train_Y:
  :param tune_X:
  :param tune_Y:
  :param goal:
  :param target_class:
  :return:
  """
  if not target_class:
    target_class = goal
  clf = learner(train_X, train_Y, tune_X, tune_Y, goal)
  tuner = DE_Tune_ML(clf, clf.get_param(), goal, target_class)
  return tuner.Tune()


def load_vec(d, data, use_pkl=False, file_name=None):
  if use_pkl:
    if os.path.isfile(file_name):
      with open(file_name, "rb") as my_pickle:
        return pickle.load(my_pickle)
  else:
    # print("call get_document_vec")
    return d.get_document_vec(data, file_name)


def print_results(clfs):
  file_name = time.strftime(os.path.sep.join([".", "results",
                                              "%Y%m%d_%H:%M:%S.txt"]))
  content = ""
  for each in clfs:
    content += each.confusion
  with open(file_name, "w") as f:
    f.write(content)
  results_process.reports(file_name)


def get_acc(cm):
  out = []
  for i in xrange(4):
    out.append(cm[i][i] / 400)
  return out


@study
def run_tuning_SVM(word2vec_src, repeats=1,
                   fold=10,
                   tuning=True):
  """
  :param word2vec_src:str, path of word2vec model
  :param repeats:int, number of repeats
  :param fold: int,number of folds
  :param tuning: boolean, tuning or not.
  :return: None
  """
  print("# word2vec:", word2vec_src)
  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
  data = PaperData(word2vec=word2vec_model)
  train_pd = load_vec(data, data.train_data, file_name=False)
  test_pd = load_vec(data, data.test_data, file_name=False)
  learner = [SK_SVM][0]
  goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
          7: "Micro_F"}[6]
  F = {}
  clfs = []
  for i in xrange(repeats):  # repeat n times here
    kf = StratifiedKFold(train_pd.loc[:, "LinkTypeId"].values, fold,
                         shuffle=True)
    for train_index, tune_index in kf:
      train_data = train_pd.ix[train_index]
      tune_data = train_pd.ix[tune_index]
      train_X = train_data.loc[:, "Output"].values
      train_Y = train_data.loc[:, "LinkTypeId"].values
      tune_X = tune_data.loc[:, "Output"].values
      tune_Y = tune_data.loc[:, "LinkTypeId"].values
      test_X = test_pd.loc[:, "Output"].values
      test_Y = test_pd.loc[:, "LinkTypeId"].values
      params, evaluation = tune_learner(learner, train_X, train_Y, tune_X,
                                        tune_Y, goal) if tuning else ({}, 0)
      clf = learner(train_X, train_Y, test_X, test_Y, goal)
      F = clf.learn(F, **params)
      clfs.append(clf)
  print_results(clfs)


@study
def run_SVM(word2vec_src):
  """
  Run SVM+word embedding experiment !
  This is the baseline method.
  :return:None
  """
  print("# word2vec:", word2vec_src)
  clf = svm.SVC(kernel="rbf", gamma=0.005)
  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
  data = PaperData(word2vec=word2vec_model)
  train_pd = load_vec(data, data.train_data, use_pkl=False)
  test_pd = load_vec(data, data.test_data, use_pkl=False)
  train_X = train_pd.loc[:, "Output"].tolist()
  train_Y = train_pd.loc[:, "LinkTypeId"].tolist()
  test_X = test_pd.loc[:, "Output"].tolist()
  test_Y = test_pd.loc[:, "LinkTypeId"].tolist()
  clf.fit(train_X, train_Y)
  predicted = clf.predict(test_X)
  print(metrics.classification_report(test_Y, predicted,
                                      labels=["1", "2", "3", "4"],
                                      # target_names=["Duplicates", "DirectLink",
                                      #               "IndirectLink",
                                      #               "Isolated"],
                                      digits=3))

  cm=metrics.confusion_matrix(test_Y, predicted, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))


def prepare_word2vec():
  print("Downloading pretrained word2vec models")
  url = "https://zenodo.org/record/807727/files/word2vecs_models.zip"
  file_name = wget.download(url)
  with zipfile.ZipFile(file_name, "r") as zip_ref:
    zip_ref.extractall()


if __name__ == "__main__":
  word_src = "word2vecs_models"
  if not os.path.exists(word_src):
    prepare_word2vec()
  elif len(os.listdir(word_src)) == 0:
    os.rmdir(word_src)
    prepare_word2vec()
  for x in xrange(10):
    random.seed(x)
    np.random.seed(x)
    myword2vecs = [os.path.join(word_src, i) for i in os.listdir(word_src)
                   if "syn" not in i]
    # run_SVM(myword2vecs[x])
    run_tuning_SVM(myword2vecs[x])