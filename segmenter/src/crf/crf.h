#ifndef __CRF_H__
#define __CRF_H__

#include <boost/program_options/variables_map.hpp>
#include "corpus.h"
#include <bits/unordered_map.h>
#include <unordered_map>
#include "cnn/cnn.h"
#include "cnn/model.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"

using namespace cnn;

struct CRFBuilder {
  CRFBuilder(cnn::Model *model, boost::program_options::variables_map conf, Corpus corpus,
                  const std::unordered_map<unsigned int, std::vector<float>> &unigram_pretrained,
                  const std::unordered_map<unsigned int, std::vector<float>> &bigram_pretrained,
                  const std::unordered_map<unsigned int, std::vector<float>> &ounigram_pretrained,
                  const std::unordered_map<unsigned int, std::vector<float>> &obigram_pretrained);

  typedef std::vector<cnn::expr::Expression> ExpressionRow;

  Expression supervised_loss(cnn::ComputationGraph *cg, const std::vector<unsigned int> &raw_unigrams,
                               const std::vector<unsigned int> &raw_bigrams, const std::vector<unsigned int> &unigrams,
                               const std::vector<unsigned int> &bigrams, const std::vector<unsigned int> &labels);

  void decode(cnn::ComputationGraph *cg, const std::vector<unsigned int> &raw_unigrams,
                const std::vector<unsigned int> &raw_bigrams, const std::vector<unsigned int> &unigrams,
                const std::vector<unsigned int> &bigrams, std::vector<unsigned int> &pred_labels);

  void set_valid_trans(const std::vector<std::string> &id2labels);

  void get_valid_labels(std::vector<unsigned int> &cur_valid_labels, unsigned int len, unsigned int cur_position,
                        std::vector<unsigned int> &pred_labels);

  // lookup parameters
  LookupParameters *p_u;
  LookupParameters *p_b;
  LookupParameters *p_pu;
  LookupParameters *p_pb;

  // lookup parameters to lstm input
  Parameters *p_pu2l;
  Parameters *p_pb2l;
  Parameters *p_u2l;
  Parameters *p_b2l;
  Parameters *p_lb;

  // bilstm
  LSTMBuilder for_lstm;
  LSTMBuilder rev_lstm;
  Parameters *p_for_guard;
  Parameters *p_rev_guard;

  // decoding layer
  LookupParameters *p_trans;
  LookupParameters *p_labels;
  Parameters *p_bif2h;
  Parameters *p_bir2h;
  Parameters *p_hb;
  Parameters *p_label2h;
  Parameters *p_h2o;
  Parameters *p_ob;

  unsigned n_labels;
  float dropout_rate;
  const std::unordered_map<unsigned, std::vector<float>>& unigram_pretrained;
  const std::unordered_map<unsigned, std::vector<float>>& bigram_pretrained;
  const std::unordered_map<unsigned, std::vector<float>>& ounigram_pretrained;
  const std::unordered_map<unsigned, std::vector<float>>& obigram_pretrained;
  std::set<unsigned> valid_trans;
  unsigned Bid, Mid, Eid, Sid;
  bool use_radical;
  Parameters *p_bi2h;
  int B;
  int M;
  int E;
  int S;
  std::vector<std::string> id2label;


  bool use_train;
  bool is_finetune;
  bool is_ofinetune;
  LookupParameters *p_pou;
  Parameters *p_pou2l;
  LookupParameters *p_pob;
  Parameters *p_pob2l;
  unsigned int UNK_IDX;
};

#endif  //  end for __CRF_H__