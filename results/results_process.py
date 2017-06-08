from __future__ import division, print_function
import pdb
import numpy as np
import pandas as pd
import os.path, sys

sys.path.append(
  os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


def reports(src,template_src=os.path.sep.join([".","results","template_tuned"])):
  his_result = get_Xu_result(template_src)

  raw = open(src, "r").readlines()
  dict_raw_results = {}
  for line in raw:
    if "400\n" in line.split(" ") or "avg" in line.split(" "):
      new_line = [i for i in line.split(" ") if len(i) >= 1]
      if new_line[0] == "avg":
        dict_raw_results["4"] =dict_raw_results.get(
          "4", []) + [[float(i) for i in new_line[3:6]]]
      else:
        dict_raw_results[str(int(new_line[0]) - 1)] = dict_raw_results.get(
          str(int(new_line[0]) - 1), []) + [[float(i) for i in new_line[1:4]]]
  my_class_report(dict_raw_results, his_result)
  print("Done!")



def get_Xu_result(template_src):
  template = open(template_src, "r").readlines()
  Xu_results = []
  for line in template:
    if len(line) > 10:
      new_line = [i for i in line.split(" ") if len(i) > 2]
      if "total" not in new_line:
        Xu_results.append(
          [float(j[5:10]) if "(" in j else j for j in new_line[1:4]])
      else:
        Xu_results.append(
          [float(j[5:10]) if "(" in j else j for j in new_line[2:5]])
  Xu_results = Xu_results[1:]  # all his paper results
  return Xu_results


def my_class_report(dict_raw_results, his_results):
  x = {}
  for key, val in dict_raw_results.iteritems():
    x[key] = np.median(val, axis=0)
  dict_results_median = [x["0"], x["1"], x["2"], x["3"]]
  p, r, f1, s = [], [], [], []
  digits = 3
  for each in dict_results_median:
    p.append(each[0])
    r.append(each[1])
    f1.append(each[2])
    s.append(400)

  labels = ["Duplicate", "Direct", "Indirect", "Isolated"]
  last_line_heading = 'avg / total'
  width = len(last_line_heading) + 5
  headers = ["precision", "recall", "f1-score", "support"]
  fmt = '%% %ds' % width  # first column: class name
  fmt += '  '
  fmt += ' '.join(['% 14s' for _ in headers])
  fmt += '\n'

  headers = [""] + headers
  report = fmt % tuple(headers)
  report += '\n'

  for i, label in enumerate(labels):
    values = [label]
    my = (p[i], r[i], f1[i])
    his = his_results[i]
    for v, h in zip(my, his):
      values += ["{0:0.{1}f}({2})".format(v, digits, h)]
    values += ["{0}".format(s[i])]
    report += fmt % tuple(values)

  report += '\n'
  # compute averages
  values = [last_line_heading]
  my_avg = (np.average(p, weights=s), np.average(r, weights=s),
            np.average(f1, weights=s))
  his_avg = his_results[-1]
  for v, h in zip(my_avg, his_avg):
    values += ["{0:0.{1}f}({2})".format(v, digits, h)]
  values += ['{0}'.format(np.sum(s))]
  report += fmt % tuple(values)
  print(report)


def get_acc(src="20170607_default_acc.csv"):
  data = pd.read_csv(src,header=None)
  pdb.set_trace()
  print(data.median())

if __name__ == "__main__":
  reports("20170322_13:45:58.txt","template_tuned")
  # reports("20170607_default.txt", "template_naive")
  # get_acc()
