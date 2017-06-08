from __future__ import division
import pdb
import os
import numpy as np
import pandas as pd
import pickle


class PaperData(object):
  def __init__(self, word2vec, dir="ASEDataset" + os.path.sep):
    self.word2vec = word2vec
    self.sentences = self.load_sentences(dir + "_4_SentenceData.txt")
    self.train_src = dir + "trainingPair.txt"
    self.test_src = dir + "testPair.txt"
    self.train_data = self.load_data(self.train_src)
    self.test_data = self.load_data(self.test_src)

  def load_data(self, dir):
    result = pd.DataFrame()
    data = pd.read_table(dir)
    i = 0
    while i < len(data):
      result = self.save(result, pd.DataFrame(
        [[data.iloc[i, 0], data.iloc[i + 1, 0], data.iloc[i + 3, 0]]],
        columns=["PostId", "RelatedPostId", "LinkTypeId"]))
      i += 5
    return result

  def load_sentences(self, dir):
    result = pd.read_table(dir, sep="\n", header=None)
    return result

  def get_document_vec(self, data, file_name=""):
    """
    this will get the word vector representations for original posts and realted
    posts, and store the results back to data, which is in pandas dataframe.

    For example, the results will have the following columns.

    "PostId","RelatedPostId","LinkType","PostIdVec","RelatedPostIdVec","Output"

    Output will be the mean of corresponding "PostIdVec" and
    "RelatedPostIdVec""

    mappings between labels and id's

    1.00              |    1     |  duplicate
    0.8               |    2     |  direct link
    0<x<0.8           |    3     |  indirect link
    0.00              |    4     |  isolated


    :param data: pandas.DataFrame of train or test data
    :return: None
    """
    if len(data) <= 0 or not isinstance(data, pd.DataFrame):
      raise ValueError(
        "Please generate {0} in pandas.dataframe first!".format(
          str(data)))
    if not self.word2vec:
      raise ValueError("Please load pre-trained word2vec mode first!")
    data["PostIdVec"] = ""
    data["RelatedPostIdVec"] = ""
    for column in ["RelatedPostId", "PostId"]:
      rows = map(int, data[column].tolist())
      pd_posts = self.sentences.iloc[rows, 0]
      for index, post_sentences in enumerate(pd_posts.tolist()):
        key_list = post_sentences.split(" ")
        x = np.array(
          [self.word2vec[i] for i in key_list if
           i in self.word2vec.vocab])
        word_count = len(x)
        word_vecs = np.sum(x, axis=0)
        data.set_value(index, column + "Vec", word_vecs / word_count)
    data["Output"] = (data["PostIdVec"] + data["RelatedPostIdVec"]) / 2
    if file_name:
      with open(file_name, "wb") as mypickle:
        pickle.dump(data, mypickle)
    return data

  def save(self, out, result):
    if out.empty:
      out = result
    else:
      out = out.append(result, ignore_index=True)
    return out

