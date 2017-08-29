#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>

struct CorpusI {
  typedef std::vector<unsigned> Sentence;
  typedef std::vector<unsigned> Output;
  typedef std::vector<std::string> RawSentence;

#ifdef _MSC_VER
  const static char* UNK;
  const static char* BAD0;
#else
  static constexpr const char* UNK = "UNK";
  static constexpr const char* BAD0 = "<BAD0>";
  static constexpr const char* BOS = "<BOS>";
#endif

  typedef std::map<std::string, unsigned> StringToIdMap;
  typedef std::map<unsigned, std::string> IdToStringMap;

  virtual ~CorpusI() {};

  bool add(const std::string& str, unsigned& max_id, StringToIdMap& str_to_id, IdToStringMap& id_to_str);
  bool find(const std::string& str, const std::vector<std::string>& id_to_label, unsigned& id) const;
  bool find(const std::string& str, const StringToIdMap& str_to_id, unsigned& id) const;
  virtual unsigned get_or_add_unigram(const std::string &unigram) = 0;
  virtual unsigned get_or_add_bigram(const std::string &bigram) = 0;
};


void load_pretrained_embedding(const std::string &embedding_file, unsigned pretrained_dim,
                               std::unordered_map<unsigned, std::vector<float> > &pretrained, CorpusI &corpus,
                               unsigned ngram);

void get_vocabulary_and_unks(std::set<unsigned> &vocabulary, std::set<unsigned> &unks, unsigned freq_limit,
                             const std::map<unsigned, std::vector<unsigned>>& sentences);

#endif  //  end for __UTILS_H__