#!/bin/env python

import os, re, numpy as np, collections
from scikits.learn.feature_extraction.text.sparse import CountVectorizer
from scikits.learn.feature_extraction.text import WordNGramAnalyzer
from scikits.learn.base import BaseEstimator
from scipy.sparse import coo_matrix
from collections import defaultdict
from pandas import *

class VowpalWabbit(BaseEstimator):
  """Handles parsing and vectorizing of vowpal wabbit data.
     Naturally handles both raw text and arbitrary features.
     Maintains fidelity of core concepts, including key meta data and feature namespaces.
     Allow for large numbers of sparse features.
     Simple support for feature pruning.
  """
  label_section_regex = re.compile("([^ ]+)(?: ([^ ]+))?(?: ([^ ]+))?")
  feature_section_regex = re.compile("([^ ]+)[ ]+(.*)")
  feature_regex = re.compile("([^:]+)(?::(.*))?")
  default_analyzer = WordNGramAnalyzer(min_n=1, max_n=1)

  def __init__(self, analyzer=default_analyzer, min_support=0.001):
    self.analyzer = analyzer
    self.min_support = min_support
    self.namespaces = set()
    self.dfs = defaultdict(lambda: defaultdict(float))
    self.index = defaultdict(lambda: defaultdict(int))

  def parse_file(self,filename):
    """low-level routine to parse vowpal wabbit data using generator
    """
    for line in open(filename):
      sections = line.strip().split('|')
      assert len(sections) >= 2, "No delimiter | found in line %s" % line
      label,importance,tag = self.label_section_regex.match(sections[0]).groups()
      features = defaultdict(lambda: defaultdict(float))

      for feature_section in sections[1:]:
        feature_section = feature_section.strip()
        assert " " in feature_section, "No space delim found in section: %s" % feature_section
        ns,feature_group = self.feature_section_regex.match(feature_section).groups()

        #simplyfing assumption: any namespace that with >= 1 valued feature does not contain text
        if ":" in feature_group:
          ns_features = feature_group.split(" ")
          for feature in ns_features:
            feature_match = self.feature_regex.match(feature)
            assert feature_match, "No features found in  %s" % feature_group
            name,value = feature_match.groups()
            value = float(value) if value != None else 1
            features[ns][name] += value
        #otherwise if no values, treat as text
        else: 
          for term in set(self.analyzer.analyze(feature_group)):
            features[ns][term] += 1

      yield label,importance,tag,features

  def flatten(self,d, parent_key='',keys_to_exclude=None):
    """Convenience method to flatten a dict"""
    items = []
    for k, v in d.items():
        if k in keys_to_exclude:
          continue
        new_key = parent_key + ':' + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(self.flatten(v, new_key,keys_to_exclude).items())
        else:
            items.append((new_key, v))

    return dict(items)

  def load_file(self,filename,ns_to_exclude_from_df = []):
    """Given filename and optionally list of namespaces to exclude from DataFrame, load file
       Return dataframe and dict of namespace->array of feature dicts
    """
    labels = []; imps = []; tags = []; features_by_ns = [];  df_features = [];
    for label, imp, tag, features in self.parse_file(filename):
      label = int(label)
      labels.append(label); 
      imps.append(float(imp)); 
      tags.append(tag)
      features_by_ns.append(features)
      features_flat = self.flatten(features,keys_to_exclude=set(ns_to_exclude_from_df))
      df_features.append(features_flat)

    df_features = DataFrame(df_features)
    two_level_cols = map(lambda flat: tuple(flat.split(":")) ,list(df_features.columns.values))
    df_features.columns = MultiIndex.from_tuples(two_level_cols, names=['ns', 'features'])

    df_meta = DataFrame({'label':labels,'imp':imps,'tag':tags})
    meta_tuples = map(lambda f: ('meta',f) ,list(df_meta.columns.values))
    df_meta.columns = MultiIndex.from_tuples(meta_tuples,names=['ns', 'features'])

    df = df_meta.join(df_features)
    return df, features_by_ns

  def fit(self,features_by_ns):
    """Given array of feature dicts, compute index and feature counts
    """
    self.namespaces = set()
    self.dfs = defaultdict(lambda: defaultdict(float))
    for features in features_by_ns:
      for ns in features.keys():
        self.namespaces.update(features.keys())
        for feature in features[ns].keys():
          self.dfs[ns][feature] += features[ns][feature]
        
    # only put supported terms in the final index
    min_df = self.min_support * len(features_by_ns) 
    self.index = defaultdict(lambda: defaultdict(int))
    for ns in self.dfs.keys():
      idx = 0
      for name, value in self.dfs[ns].iteritems():
        if value >= min_df: 
          self.index[ns][name] = idx
          idx += 1

  def fit_transform(self,features_by_ns):
    self.fit(features_by_ns)
    return self.transform(features_by_ns)

  def transform(self,features_by_ns):
    """Given array of feature counts, return dict of ns->sparse feature matrix
    """
    assert len(self.index) > 0, 'No index. Call fit() first.'
    row_ids_by_ns = defaultdict(list)
    feature_ids_by_ns = defaultdict(list)
    values_by_ns = defaultdict(list)
    row_id = 0

    # build large sparse triples of row id, feature id, & value 
    for features_by_ns in features_by_ns:
      for ns in features_by_ns.keys():
        features = features_by_ns[ns]
        for feature in features.keys():
          feature_idx = self.index[ns].get(feature)
          if feature_idx:
            row_ids_by_ns[ns].append(row_id)
            feature_ids_by_ns[ns].append(feature_idx)
            values_by_ns[ns].append(features[feature])
      row_id += 1

    matrix_by_ns = {}
    for ns in self.namespaces:
      idxs = (row_ids_by_ns[ns], feature_ids_by_ns[ns])
      num_cols = len(self.index[ns])
      matrix_by_ns[ns] = coo_matrix((values_by_ns[ns], idxs), shape=(row_id, num_cols), dtype=np.float32)  

    return matrix_by_ns
