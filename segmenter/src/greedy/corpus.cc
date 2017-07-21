#include "utils.h"
#include "logging.h"
#include "corpus.h"
#include <boost/assert.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <cassert>


Corpus::Corpus()
  : max_unigram(0), max_bigram(0){
}


// TODO add radical and strokes
void Corpus::load_training_data(const std::string &filename, const std::string &mode) {
  std::ifstream train_file(filename);
  BOOST_ASSERT_MSG(train_file, "Failed to open training file.");
  std::string line, prev_unigram;

  // add UNK
  if (mode == "train"){
    BOOST_ASSERT_MSG(max_bigram==0 && max_unigram==0, "max unigram and bigram should be 0");
    unigram2id[Corpus::UNK] = 0, id2unigram[0] = Corpus::UNK;
    bigram2id[Corpus::UNK] = 0, id2bigram[0] = Corpus::UNK;
    radical2id[Corpus::UNK] = 0, id2radical[0] = Corpus::UNK;
    max_unigram = 1, max_bigram = 1, max_radical = 1;
  }
  else{
    BOOST_ASSERT(max_unigram >= 1);
    BOOST_ASSERT(max_bigram >= 1);
    BOOST_ASSERT(max_radical >= 1);
    BOOST_ASSERT(id2label.size() == 4);
  }

  std::vector<unsigned> cur_unigrams;
  std::vector<unsigned> cur_bigrams;
  std::vector<unsigned> cur_radicals;
  std::vector<unsigned> cur_labels;

  unsigned sid = 0;
  while (std::getline(train_file, line)){
    if (line.empty()){
      if (cur_unigrams.size() == 0){
        continue;
      }
      if (mode == "train"){
        train_unigram_sentences[sid] = cur_unigrams;
        train_bigram_sentences[sid] = cur_bigrams;
        train_radical_sentences[sid] = cur_radicals;
        train_labels[sid] = cur_labels;
        sid ++;
        n_train = sid;
      }
      else if(mode == "dev"){
        dev_unigram_sentences[sid] = cur_unigrams;
        dev_bigram_sentences[sid] = cur_bigrams;
        dev_radical_sentences[sid] = cur_radicals;
        dev_labels[sid] = cur_labels;
        sid ++;
        n_dev = sid;
      }
      else if(mode == "test"){
        test_unigram_sentences[sid] = cur_unigrams;
        test_bigram_sentences[sid] = cur_bigrams;
        test_radical_sentences[sid] = cur_radicals;
        test_labels[sid] = cur_labels;
        sid ++;
        n_test = sid;
      }
      else{
        BOOST_ASSERT_MSG(false, "wrong mode of training data");
      }

      cur_unigrams.clear();
      cur_bigrams.clear();
      cur_labels.clear();
      cur_radicals.clear();
    }
    else{
      std::vector<std::string> items;

      boost::algorithm::trim(line);
      boost::algorithm::split(items, line, boost::is_any_of("\t"), boost::token_compress_on);

      BOOST_ASSERT_MSG(items.size() >= 2, "Ill format of training data");
      std::string& unigram = items[0];
      std::string& radical = items[1];
      std::string& label = items.back();

      std::string bigram;
      if (cur_unigrams.size() == 0) {
        bigram = Corpus::BOS + unigram;
      }
      else{
        bigram = prev_unigram + unigram;
      }
      prev_unigram = unigram;

      if (mode == "train"){
        add(unigram, max_unigram, unigram2id, id2unigram);
        add(bigram, max_bigram, bigram2id, id2bigram);
        add(radical, max_radical, radical2id, id2radical);
        unsigned lid;
        bool found = find(label, id2label, lid);
        if (!found){
          id2label.push_back(label);
          lid = id2label.size() - 1;
        }

        cur_unigrams.push_back(unigram2id[unigram]);
        cur_bigrams.push_back(bigram2id[bigram]);
        cur_radicals.push_back(radical2id[radical]);
        cur_labels.push_back(lid);
      }
      else{
        unsigned payload = 0;
        if (!find(unigram, unigram2id, payload)){
          find(Corpus::UNK, unigram2id , payload);
          cur_unigrams.push_back(payload);
        }
        else{
          cur_unigrams.push_back(payload);
        }

        if(!find(bigram, bigram2id, payload)){
          find(Corpus::UNK, bigram2id, payload);
          cur_bigrams.push_back(payload);
        }
        else{
          cur_bigrams.push_back(payload);
        }

        if(!find(radical, radical2id, payload)){
          find(Corpus::UNK, radical2id, payload);
          cur_radicals.push_back(payload);
        }
        else{
          cur_radicals.push_back(payload);
        }

        bool found = find(label, id2label, payload);
        if (!found) {
          BOOST_ASSERT_MSG(false, "Unknown label in development data.");
        }

        cur_labels.push_back(payload);
      }
    }
  }

  if (mode == "train") { stat(); }
  train_file.close();
}

void Corpus::stat() const {
  //_INFO << "action indexing ...";
  //for (auto l : id2label) {
  //  _INFO << l;
  //}
  _INFO << "max id of unigrams: " << max_unigram;
  _INFO << "max id of bigrams: " << max_bigram;
  _INFO << "max id of radicals" << max_radical;
  _INFO << "number of labels: " << id2label.size();

  _INFO << "label indexing ...";
  unsigned index = 0;
  for (auto& p : id2label) { _INFO << index << " : " << p; index += 1; }
}

unsigned Corpus::get_or_add_unigram(const std::string& word) {
  unsigned payload;
  if (!find(word, unigram2id, payload)) {
    add(word, max_unigram, unigram2id, id2unigram);
    return unigram2id[word];
  }
  return payload;
}

unsigned Corpus::get_or_add_bigram(const std::string &bigram) {
  unsigned payload;
  if (!find(bigram, bigram2id, payload)) {
    add(bigram, max_bigram, bigram2id, id2bigram);
    return bigram2id[bigram];
  }
  return payload;
}

void Corpus::load_test_data(const std::string &filename) {
  std::ifstream test_file(filename);
  BOOST_ASSERT_MSG(test_file, "Failed to open test file.");
  std::string line, prev_unigram;

  BOOST_ASSERT(max_unigram >= 1);
  BOOST_ASSERT(max_bigram >= 1);
  BOOST_ASSERT(id2label.size() >= 2);

  std::vector<unsigned> cur_unigrams;
  std::vector<unsigned> cur_bigrams;
  std::vector<unsigned> cur_radicals;
  std::vector<std::string> cur_unigrams_str;

  unsigned sid = 0;
  while (std::getline(test_file, line)){
    if (line.empty()){
      if (cur_unigrams.size() == 0){
        continue;
      }
      test_unigram_sentences[sid] = cur_unigrams;
      test_bigram_sentences[sid] = cur_bigrams;
      test_radical_sentences[sid] = cur_radicals;
      test_unigram_sentences_str[sid] = cur_unigrams_str;

      sid ++;
      n_test = sid;
      cur_unigrams.clear();
      cur_bigrams.clear();
      cur_radicals.clear();
      cur_unigrams_str.clear();
    }
    else{
      std::vector<std::string> items;

      boost::algorithm::trim(line);
      boost::algorithm::split(items, line, boost::is_any_of("\t"), boost::token_compress_on);

      BOOST_ASSERT_MSG(items.size() >= 1, "Ill format of training data");
      std::string& unigram = items[0];
      std::string& radical = items[1];
      cur_unigrams_str.push_back(unigram);

      std::string bigram;
      if (cur_unigrams.size() == 0) {
        bigram = Corpus::BOS + unigram;
      }
      else{
        bigram = prev_unigram + unigram;
      }
      prev_unigram = unigram;

      unsigned payload = 0;
      if (!find(unigram, unigram2id, payload)){
        find(Corpus::UNK, unigram2id, payload);
        cur_unigrams.push_back(payload);
      }
      else{
        cur_unigrams.push_back(payload);
      }

      if(!find(bigram, bigram2id, payload)){
        find(Corpus::UNK, bigram2id, payload);
        cur_bigrams.push_back(payload);
      }
      else{
        cur_bigrams.push_back(payload);
      }

      if(!find(radical, radical2id, payload)){
        find(Corpus::UNK, radical2id, payload);
        cur_radicals.push_back(payload);
      }
      else{
        cur_radicals.push_back(payload);
      }
    }
  }

  test_file.close();
}
