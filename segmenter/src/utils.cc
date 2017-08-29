#include <iostream>
#include <fstream>
#include <sstream>
#include "utils.h"
#include "logging.h"

#if _MSC_VER
const char* CorpusI::UNK = "UNK";
const char* CorpusI::BAD0 = "<BAD0>";
#endif


bool CorpusI::add(const std::string& str, unsigned& max_id,
  CorpusI::StringToIdMap& str_to_id, CorpusI::IdToStringMap& id_to_str) {
  if (str_to_id.find(str) == str_to_id.end()) {
    str_to_id[str] = max_id;
    id_to_str[max_id] = str;
    ++max_id;
    return true;
  }
  return false;
}


bool CorpusI::find(const std::string& str, const std::vector<std::string>& id_to_label,
  unsigned& id) const {
  for (unsigned i = 0; i < id_to_label.size(); ++i) {
    auto l = id_to_label[i];
    if (l == str) {
      id = i;
      return true;
    }
  }
  id = id_to_label.size();
  return false;
}

bool CorpusI::find(const std::string& str, const CorpusI::StringToIdMap& str_to_id,
  unsigned& id) const {
  auto result = str_to_id.find(str);
  if (result != str_to_id.end()) {
    id = result->second;
    return true;
  }
  id = 0;
  return false;
}


//  TODO cut the pretrained embedding
void load_pretrained_embedding(const std::string &embedding_file, unsigned pretrained_dim,
                               std::unordered_map<unsigned, std::vector<float> > &pretrained, CorpusI &corpus,
                               unsigned ngram) {
  // Loading the pretrained word embedding from file. Embedding file is presented
  // line by line, with the lexical word in leading place and 50 real number follows.  
  // pretrained[corpus.get_or_add_unigram(CorpusI::UNK)] = std::vector<float>(pretrained_dim, 0);
  std::vector<float> unk_vector(pretrained_dim, 0); // average all vectors as UNK embedding
  _INFO << "Loading from " << embedding_file << " with " << pretrained_dim << " dimensions";
  std::ifstream in(embedding_file);
  std::string line;
  // skip the first line
  std::getline(in, line);
  std::vector<float> v(pretrained_dim, 0);
  std::string word;
  double n_vecotrs = 0;
  while (std::getline(in, line)) {
    std::istringstream iss(line);
    n_vecotrs += 1;
    iss >> word;
    for (unsigned i = 0; i < pretrained_dim; ++i) { iss >> v[i]; unk_vector[i] += v[i]; }
    unsigned id;
    if (ngram == 1)
      id = corpus.get_or_add_unigram(word);
    else if(ngram == 2)
      id = corpus.get_or_add_bigram(word);
    pretrained[id] = v;
  }
  for (unsigned i=0; i<pretrained_dim; i++) { unk_vector[i] /= n_vecotrs; }
  pretrained[corpus.get_or_add_unigram(CorpusI::UNK)] = unk_vector;

}

void get_vocabulary_and_unks(std::set<unsigned> &vocabulary, std::set<unsigned> &unks, unsigned freq_limit, const std::map<unsigned, std::vector<unsigned>>& sentences) {
  std::map<unsigned, unsigned> counter;

  for (auto& payload : sentences) {
    for (auto& word : payload.second) {
      vocabulary.insert(word);
      ++counter[word];
    }
  }
  for (auto& payload : counter) {
    if (payload.second <= freq_limit) { unks.insert(payload.first); }
  }
}

