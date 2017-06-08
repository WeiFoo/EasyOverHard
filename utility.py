from __future__ import print_function, division
import datetime
import sys
import re
import time
import pdb
import random


def study(f):
  def wrapper(*lst):
    # rseed() # reset the seed to our default
    what = f.__name__  # print the function name
    doc = f.__doc__  # print the function doc
    if doc:
      doc = re.sub(r"\n[ \t]*", "\n# ", doc)
    # print when this ran
    show = datetime.datetime.now().strftime
    print("\n###", what, "#" * 50)
    print("#", show("%Y-%m-%d %H:%M:%S"))
    if doc: print("#", doc)
    t1 = time.time()
    f(*lst)  # run the function
    t2 = time.time()  # show how long it took to run
    print("\n" + ("-" * 72))
    # showd(The)       # print the options
    print("# Runtime: %.3f secs" % (t2 - t1))

  return wrapper


### Coercion  #####################################
def atom(x):
  try:
    return int(x)
  except ValueError:
    try:
      return float(x)
    except ValueError:
      return x


### Command line processing ########################
def cmd(com="life(seed=1)"):
  "Convert command line to a function call."
  if len(sys.argv) < 2:
    return

  def strp(x): return isinstance(x, basestring)

  def wrap(x): return "'%s'" % x if strp(x) else str(x)

  words = map(wrap, map(atom, sys.argv[2:]))
  return sys.argv[1] + '(' + ','.join(words) + ')'
