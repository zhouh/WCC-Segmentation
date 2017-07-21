#ifndef __CPYPDICT_H__
#define __CPYPDICT_H__

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <map>
#include <vector>
#include <functional>
#include "utils.h"

struct Corpus: public CorpusI {
  std::map<unsigned, std::vector<unsigned>> train_unigram_sentences;
  std::map<unsigned, std::vector<unsigned>> train_bigram_sentences;
  std::map<unsigned, std::vector<unsigned>> train_radical_sentences;
  std::map<unsigned, std::vector<unsigned>> train_labels;

  std::map<unsigned, std::vector<unsigned>> dev_unigram_sentences;
  std::map<unsigned, std::vector<unsigned>> dev_bigram_sentences;
  std::map<unsigned, std::vector<unsigned>> dev_radical_sentences;
  std::map<unsigned, std::vector<unsigned>> dev_labels;

  std::map<unsigned, std::vector<unsigned>> test_unigram_sentences;
  std::map<unsigned, std::vector<unsigned>> test_bigram_sentences;
  std::map<unsigned, std::vector<unsigned>> test_radical_sentences;
  std::map<unsigned, std::vector<unsigned>> test_labels;
  std::map<unsigned, std::vector<std::string>> test_unigram_sentences_str;

  unsigned n_train; /* number of sentences in the training data */
  unsigned n_dev; /* number of sentences in the development data */
  unsigned n_test;

  unsigned n_labels;

  unsigned max_unigram;
  unsigned max_bigram;
  unsigned max_radical;

  StringToIdMap unigram2id;
  IdToStringMap id2unigram;

  StringToIdMap bigram2id;
  IdToStringMap id2bigram;

  StringToIdMap radical2id;
  IdToStringMap id2radical;

  std::vector<std::string> id2label;

  Corpus();
  
  void stat() const;

  unsigned get_or_add_unigram(const std::string& unigram);
  unsigned get_or_add_bigram(const std::string& bigram);

  void load_training_data(const std::string &filename, const std::string &mode);
  void load_test_data(const std::string &filename);
};

#endif  //  end for __CPYPDICT_H__
