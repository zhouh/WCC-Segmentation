#include <iostream>
#include <map>
#include <chrono>
#include <ctime>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <cassert>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "logging.h"
#include "training_utils.h"
#include "crf.h"
#include "../corpus.h"
#include "glog/logging.h"
#include "eval.h"

namespace po = boost::program_options;

Corpus corpus;

void print_conf(const boost::program_options::variables_map &conf);

void init_command_line(int argc, char* argv[], po::variables_map* conf) {
  po::options_description opts("LSTM-GREEDY");
  opts.add_options()
    ("is_train,i", po::value<unsigned>(), "train or test")
    ("optimizer", po::value<std::string>()->default_value("simple_sgd"), "The optimizer.")
    ("model_dir", po::value<std::string>()->default_value("model/"), "Model dir")
    ("train,T", po::value<std::string>(), "The path to the train data.")
    ("dev,d", po::value<std::string>(), "The path to the dev data")
    ("test,t", po::value<std::string>(), "The path to the test data")
    ("model,m", po::value<std::string>(), "The path to the model, used for test time.")
    ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy.")
    ("unk_prob,u", po::value<double>()->default_value(0.2), "Probabilty with which to replace singletons.")
    ("layers", po::value<unsigned>()->default_value(1), "number of LSTM layers")
    ("unigram, c", po::value<std::string>(), "The path to the in domian unigram embedding.")
    ("bigram, b", po::value<std::string>(), "The path to the in domain bigram embedding")
    ("ounigram, oc", po::value<std::string>(), "The path to the out of domian unigram embedding.")
    ("obigram, ob", po::value<std::string>(), "The path to the out of domain bigram embedding")
    ("unigram_dim", po::value<unsigned>()->default_value(50), "unigram embedding dim")
    ("bigram_dim", po::value<unsigned>()->default_value(50), "bigram embedding dim")
    ("label_dim", po::value<unsigned>()->default_value(32), "label embedding dim")
    ("lstm_hidden_dim", po::value<unsigned>()->default_value(100), "LSTM hidden dimension")
    ("lstm_input_dim", po::value<unsigned>()->default_value(100), "LSTM input dimension")
    ("hidden_dim", po::value<unsigned>()->default_value(100), "hidden dim")
    ("maxiter", po::value<unsigned>()->default_value(30), "Max number of iterations.")
    ("dropout", po::value<float>()->default_value(0.3), "the dropout rate")
    ("report_stops", po::value<unsigned>()->default_value(100), "the number of stops for reporting.")
    ("evaluate_stops", po::value<unsigned>()->default_value(2500), "the number of stops for evaluation.")
    ("outfile", po::value<std::string>(), "outputfile in test mode")
    ("use_train", po::value<unsigned>()->default_value(1), "whether to use emb just in training data")
    ("finetune", po::value<unsigned>()->default_value(0), "whether to finetune the pretrained in-domain emb")
    ("ofinetune", po::value<unsigned>()->default_value(0), "whether to finetune the pretrained out-of-domain emb")
    ("help,h", "Show help information");

  po::store(po::parse_command_line(argc, argv, opts), *conf);
  if (conf->count("help")) {
    std::cerr << opts << std::endl;
    exit(1);
  }

  init_boost_log(true);
  if (conf->count("train") == 0) {
    std::cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }

}

std::string get_current_time(){
  time_t rawtime;
  struct tm* timeinfo;
  char buffer[80];
  time (&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer,80,"%d-%m-%Y %I:%M",timeinfo);
  std::string time_str(buffer);
  return time_str;
}


// TODO specify model with prefix
std::string get_model_name(const po::variables_map& conf){
  std::ostringstream os;
  if (conf.count("model")) {
    os << conf["model"].as<std::string>();
  }else{
    os << conf["model_dir"].as<std::string>()
       <<"model_"
       <<conf["bigram_dim"].as<unsigned>()
       <<"_" << get_current_time()
       <<".bin";
  }
  return  os.str();
}

double evaluate(Corpus &corpus, CRFBuilder &engine, const std::set<unsigned> &training_unigram_vocab,
                const std::set<unsigned> &training_bigram_vocab, std::string mode, const std::string &outfile) {
  std::ofstream ofs;
  if (mode == "test") {
    ofs.open(outfile);
    BOOST_ASSERT(ofs != NULL);
  }

  auto kUNK = corpus.get_or_add_unigram(Corpus::UNK);
  double n_total_gold_words = 0;
  double n_total_pred_words = 0;
  double n_total_right_words = 0;

  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned n;
  if (mode == "dev")
    n = corpus.n_dev;
  else if (mode == "test")
    n = corpus.n_test;
  else
    BOOST_ASSERT_MSG(false, "wrong mode in evaluation");

  for (unsigned sid = 0; sid < n; ++sid) {
    std::vector<unsigned> raw_unigrams, raw_bigrams, labels;
    std::vector<std::string> unigrams_str;
    if (mode == "dev"){
      raw_unigrams = corpus.dev_unigram_sentences[sid];
      raw_bigrams = corpus.dev_bigram_sentences[sid];
      labels = corpus.dev_labels[sid];
    }
    else if(mode == "test"){
      raw_unigrams = corpus.test_unigram_sentences[sid];
      raw_bigrams = corpus.test_bigram_sentences[sid];
      labels = corpus.test_labels[sid];
      unigrams_str = corpus.test_unigram_sentences_str[sid];
    }

    unsigned len = static_cast<int>(raw_unigrams.size());

    std::vector<unsigned> unigrams = raw_unigrams;
    for (auto& w : unigrams) {
      if (training_unigram_vocab.count(w) == 0) w = kUNK;
    }
    std::vector<unsigned> bigrams = raw_bigrams;
    for (auto &w : bigrams) {
      if (training_bigram_vocab.count(w) == 0) w = kUNK;
    }

    BOOST_ASSERT_MSG(len == labels.size(), "Unequal sentence and gold label length");

    cnn::ComputationGraph cg;
    std::vector<unsigned> pred_labels;
    engine.decode(&cg, raw_unigrams, raw_bigrams, unigrams, bigrams, pred_labels);

    //_INFO << "eval sent " << sid;
    //for(auto &l : labels)
    //  std::cerr << corpus.id2label[l] << " ";
    //std::cerr << std::endl;
    //for(auto &l : pred_labels)
    //  std::cerr << corpus.id2label[l] << " ";
    //std::cerr << std::endl;

    BOOST_ASSERT_MSG(len == pred_labels.size(), "Unequal sentence and predict label length");
    if (mode == "test"){
      for (unsigned i=0; i<len; i++){
      ofs << unigrams_str[i] << "\t" << corpus.id2label[pred_labels[i]] << std::endl;
      }
      ofs << std::endl;
    }

    std::tuple<unsigned, unsigned, unsigned> result = eval_sentence(pred_labels, labels, corpus.id2label);
    n_total_gold_words += std::get<0>(result);
    n_total_pred_words += std::get<1>(result);
    n_total_right_words += std::get<2>(result);
  }

  double precision = n_total_right_words / n_total_pred_words;
  double recall = n_total_right_words / n_total_gold_words;
  double f_score = 2 * precision * recall / (precision + recall);
  auto t_end = std::chrono::high_resolution_clock::now();
  _INFO << mode <<" Test f-score: " << f_score <<
        " [" << n <<
        " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
  return f_score;
}

void train(boost::program_options::variables_map conf, cnn::Model &model, const std::string &model_name, Corpus &corpus,
           const std::set<unsigned int> &training_unigram_vocab, const std::set<unsigned int> &training_bigram_vocab,
           const std::set<unsigned int> &uni_unks, const std::set<unsigned int> &bi_unks, CRFBuilder &engine,
           const std::string &outfile) {

  _INFO << "start training ...";

  // Setup the trainer. sgd or adadelta ...
  cnn::Trainer* trainer = get_trainer(conf, &model);
  //std::string model_name = get_model_name(conf);

  // Order for shuffle.
  std::vector<unsigned> order(corpus.n_train);
  for (unsigned i = 0; i < corpus.n_train; ++i) { order[i] = i; }

  unsigned kUNK = corpus.get_or_add_unigram(Corpus::UNK);
  auto maxiter = conf["maxiter"].as<unsigned>();
  double n_seen = 0;
  double n_corr_tokens = 0, n_tokens = 0, llh = 0;
  double batchly_n_corr_tokens = 0, batchly_n_tokens = 0, batchly_llh = 0;
  /* # correct tokens, # tokens, # sentence, loglikelihood.*/

  int logc = 0;
  _INFO << "number of training instances: " << corpus.n_train;
  _INFO << "going to train " << maxiter << " iterations.";

  auto unk_strategy = conf["unk_strategy"].as<unsigned>();
  auto unk_prob = conf["unk_prob"].as<double>();
  double best_f_score = 0.;

  for (unsigned iter = 0; iter < maxiter; ++iter) {
    _INFO << "start of iteration #" << iter << ", training data is shuffled.";
    std::shuffle(order.begin(), order.end(), (*cnn::rndeng));

    for (unsigned i = 0; i < order.size(); ++i) {
      auto sid = order[i];
      const std::vector<unsigned>& raw_unigrams = corpus.train_unigram_sentences[sid];
      const std::vector<unsigned>& raw_bigrams = corpus.train_bigram_sentences[sid];
      const std::vector<unsigned>& labels = corpus.train_labels[sid];

      std::vector<unsigned> unigrams = raw_unigrams;
      std::vector<unsigned> bigrams = raw_bigrams;
      if (unk_strategy == 1) {
        for (auto& w : unigrams) {
          if (uni_unks.count(w) && cnn::rand01() < unk_prob) { w = kUNK; }
        }
        for (auto& w : bigrams) {
          if (bi_unks.count(w) && cnn::rand01() < unk_prob) { w = kUNK; }
        }
      }

      double lp;
      {
        cnn::ComputationGraph hg;
        //_INFO << "sent " << sid << " starts to train";
        engine.supervised_loss(&hg, raw_unigrams, raw_bigrams, unigrams, bigrams, labels);
        //_INFO << "sent " << sid << " finsihed train";
        lp = cnn::as_scalar(hg.incremental_forward());
        BOOST_ASSERT_MSG(lp >= 0, "Log prob < 0 on sentence");
        hg.backward();
        trainer->update(1.);
      }

      llh += lp; batchly_llh += lp;
      n_seen += 1;
      ++logc;

      LOG_EVERY_N(INFO, 1000) << "Iter iter # " << iter << " , " << logc % corpus.n_train << " sents are trained";

      n_tokens += labels.size(); batchly_n_tokens += labels.size();

      if (logc % conf["report_stops"].as<unsigned>() == 0) {
        trainer->status();
        _INFO << "iter (batch) #" << iter << " (epoch " << n_seen / corpus.n_train
              << ") llh: " << batchly_llh << " ppl: " << exp(batchly_llh / batchly_n_tokens);
        n_corr_tokens += batchly_n_corr_tokens;
        batchly_llh = batchly_n_tokens = batchly_n_corr_tokens = 0.;
      }

      if (logc % conf["evaluate_stops"].as<unsigned>() == 0) {
        double f_score = evaluate(corpus, engine, training_unigram_vocab, training_bigram_vocab, "dev", outfile);
        if (f_score > best_f_score) {
          best_f_score = f_score;
          _INFO << "new best dev " << best_f_score << " is achieved, model updated.";
          LOG(INFO) << "Iter # " << iter <<", new best dev " << best_f_score << " is achieved, model updated.";
          if (conf.count("test")){
            double test_score = evaluate(corpus, engine, training_unigram_vocab, training_bigram_vocab, "test", outfile);
            //test(corpus, engine, training_unigram_vocab, training_bigram_vocab, outfile);
            _INFO << "new test score " << test_score << " is achieved, model updated.";
            LOG(INFO) << "Iter # " << iter <<", new test score " << test_score << " is achieved, model updated.";
          }
          std::ofstream out(model_name);
          boost::archive::text_oarchive oa(out);
          oa << model;
        }
      }
    }
    _INFO << "iteration #" << iter + 1 << " (epoch " << n_seen / corpus.n_train
          << ") llh: " << llh << " ppl: " << exp(llh / n_tokens);
    llh = n_tokens = n_corr_tokens = 0.;

    double f_score = evaluate(corpus, engine, training_unigram_vocab, training_bigram_vocab, "dev", outfile);
    if (f_score > best_f_score) {
      best_f_score = f_score;
      _INFO << "new best dev " << best_f_score << " is achieved, model updated.";
      LOG(INFO) << "Iter # " << iter <<", new best dev " << best_f_score << " is achieved, model updated.";
      if (conf.count("test")){
        double test_score = evaluate(corpus, engine, training_unigram_vocab, training_bigram_vocab, "test", outfile);
        //test(corpus, engine, training_unigram_vocab, training_bigram_vocab, outfile);
        _INFO << "new test score " << test_score << " is achieved, model updated.";
        LOG(INFO) << "Iter # " << iter <<", new test score " << test_score << " is achieved, model updated.";
      }

      std::ofstream out(model_name);
      boost::archive::text_oarchive oa(out);
      oa << model;
    }
    if (conf["optimizer"].as<std::string>() == "simple_sgd" || conf["optimizer"].as<std::string>() == "momentum_sgd") {
      trainer->update_epoch();
    }
  }
  delete trainer;
}

void test(Corpus &corpus, CRFBuilder &engine, const std::set<unsigned int>& training_unigram_vocab,
            const std::set<unsigned int> &training_bigram_vocab, const std::string &outfile) {

  std::ofstream ofs(outfile);
  BOOST_ASSERT(ofs != NULL);

  auto kUNK = corpus.get_or_add_unigram(Corpus::UNK);
  double n_total_gold_words = 0;
  double n_total_pred_words = 0;
  double n_total_right_words = 0;

  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned n = corpus.n_test;

  for (unsigned sid = 0; sid < n; ++sid) {
    std::vector<unsigned> raw_unigrams, raw_bigrams, labels;
    std::vector<std::string> unigrams_str;
    raw_unigrams = corpus.test_unigram_sentences[sid];
    raw_bigrams = corpus.test_bigram_sentences[sid];
    unigrams_str = corpus.test_unigram_sentences_str[sid];
    labels = corpus.test_labels[sid];

    unsigned len = raw_unigrams.size();

    std::vector<unsigned> unigrams = raw_unigrams;
    for (auto& w : unigrams) {
      if (training_unigram_vocab.count(w) == 0) w = kUNK;
    }
    std::vector<unsigned> bigrams = raw_bigrams;
    for (auto &w : bigrams) {
      if (training_bigram_vocab.count(w) == 0) w = kUNK;
    }

    //BOOST_ASSERT_MSG(len == labels.size(), "Unequal sentence and gold label length");

    cnn::ComputationGraph cg;
    std::vector<unsigned> pred_labels;
    //_INFO << "sent " << sid << " starts decoding";
    engine.decode(&cg, raw_unigrams, raw_bigrams, unigrams, bigrams, pred_labels);
    BOOST_ASSERT_MSG(len == pred_labels.size(), "Unequal sentence and predict label length");

    std::tuple<unsigned, unsigned, unsigned> result = eval_sentence(pred_labels, labels, corpus.id2label);
    n_total_gold_words += std::get<0>(result);
    n_total_pred_words += std::get<1>(result);
    n_total_right_words += std::get<2>(result);

    // output as col like file
    for (unsigned i=0; i<len; i++){
      ofs << unigrams_str[i] << "\t" << corpus.id2label[pred_labels[i]] << std::endl;
    }
    ofs << std::endl;
  }

  double precision = n_total_right_words / n_total_pred_words;
  double recall = n_total_right_words / n_total_gold_words;
  double f_score = 2 * precision * recall / (precision + recall);
  _INFO << "test fscore" << f_score;
  ofs.close();
  auto t_end = std::chrono::high_resolution_clock::now();
  _INFO << " [" << n <<
        " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
}

void segment(Corpus &corpus, CRFBuilder &engine, const std::set<unsigned int>& training_unigram_vocab,
          const std::set<unsigned int> &training_bigram_vocab, const std::string &outfile) {

  std::ofstream ofs(outfile);
  BOOST_ASSERT(ofs != NULL);

  auto kUNK = corpus.get_or_add_unigram(Corpus::UNK);
  double n_total_gold_words = 0;
  double n_total_pred_words = 0;
  double n_total_right_words = 0;

  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned n = corpus.n_test;

  for (unsigned sid = 0; sid < n; ++sid) {
    std::vector<unsigned> raw_unigrams, raw_bigrams;
    std::vector<std::string> unigrams_str;
    raw_unigrams = corpus.test_unigram_sentences[sid];
    raw_bigrams = corpus.test_bigram_sentences[sid];
    unigrams_str = corpus.test_unigram_sentences_str[sid];

    unsigned len = raw_unigrams.size();

    std::vector<unsigned> unigrams = raw_unigrams;
    for (auto& w : unigrams) {
      if (training_unigram_vocab.count(w) == 0) w = kUNK;
    }
    std::vector<unsigned> bigrams = raw_bigrams;
    for (auto &w : bigrams) {
      if (training_bigram_vocab.count(w) == 0) w = kUNK;
    }

    //BOOST_ASSERT_MSG(len == labels.size(), "Unequal sentence and gold label length");

    cnn::ComputationGraph cg;
    std::vector<unsigned> pred_labels;
    //_INFO << "sent " << sid << " starts decoding";
    engine.decode(&cg, raw_unigrams, raw_bigrams, unigrams, bigrams, pred_labels);
    BOOST_ASSERT_MSG(len == pred_labels.size(), "Unequal sentence and predict label length");

    // output as col like file
    for (unsigned i=0; i<len; i++){
      ofs << unigrams_str[i] << "\t" << corpus.id2label[pred_labels[i]] << std::endl;
    }
    ofs << std::endl;
  }

  ofs.close();
  auto t_end = std::chrono::high_resolution_clock::now();
  _INFO << " [" << n <<
        " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
}

void print_conf(const boost::program_options::variables_map &conf) {
  LOG(INFO) << "optimizer: " << conf["optimizer"].as<std::string>();
  LOG_IF(INFO, conf.count("test")) << "test data: " << conf["test"].as<std::string>();
  LOG(INFO) << "unigram dim: " << conf["unigram_dim"].as<unsigned>();
  LOG_IF(INFO, conf.count("unigram")) << "pretraiend unigram embedding: " << conf["unigram"].as<std::string>();
  LOG(INFO) << "bigram dim: " << conf["bigram_dim"].as<unsigned>();
  LOG_IF(INFO, conf.count("bigram")) << "pretrained bigram embedding: " << conf["bigram"].as<std::string>();
  LOG(INFO) << "label dim: " << conf["label_dim"].as<unsigned>();
  LOG(INFO) << "dropout rate: " << conf["dropout"].as<float>();
  LOG(INFO) << "train data" << conf["train"].as<std::string>();
  LOG(INFO) << "dev data: " << conf["dev"].as<std::string>();
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "glog starts";

  cnn::Initialize(argc, argv, 123);

  std::cerr << "command:";
  for (int i = 0; i < argc; ++i) { std::cerr << ' ' << argv[i]; }
  std::cerr << std::endl;

  po::variables_map conf; // configuration map
  init_command_line(argc, argv, &conf);

  if (conf["unk_strategy"].as<unsigned>() == 1) {
    _INFO << "unknown word strategy: STOCHASTIC REPLACEMENT";
  } else {
    _INFO << "unknown word strategy: NO REPLACEMENT";
  }

  std::string model_name;
  if (conf.count("train")) {
    model_name = get_model_name(conf);
    _INFO << "going to write parameters to file: " << model_name;
  } else {
    model_name = conf["model"].as<std::string>();
    _INFO << "going to load parameters from file: " << model_name;
  }

  bool is_train = (conf["is_train"].as<unsigned>() == 1);
  if (is_train) print_conf(conf);

  _INFO << "load training data";
  corpus.load_training_data(conf["train"].as<std::string>(), "train");

  std::set<unsigned> training_unigram_vocab, uni_unks;
  std::set<unsigned> training_bigram_vocab, bi_unks;

  // TODO set freq
  get_vocabulary_and_unks(training_unigram_vocab, uni_unks, 2, corpus.train_unigram_sentences);
  get_vocabulary_and_unks(training_bigram_vocab, bi_unks, 1, corpus.train_bigram_sentences);

  // pretrained emb
  _INFO << "set low freq chars as UNK";
  std::unordered_map<unsigned, std::vector<float>> unigram_pretrained;
  std::unordered_map<unsigned, std::vector<float>> bigram_pretrained;
  std::unordered_map<unsigned, std::vector<float>> ounigram_pretrained;
  std::unordered_map<unsigned, std::vector<float>> obigram_pretrained;

  if (conf.count("unigram")) {
    load_pretrained_embedding(conf["unigram"].as<std::string>(), conf["unigram_dim"].as<unsigned>(), unigram_pretrained,
                              corpus, 1);
    _INFO << unigram_pretrained.size() <<" pretrained unigram embedding is loaded.";
  }

  if (conf.count("bigram")) {
    load_pretrained_embedding(conf["bigram"].as<std::string>(), conf["bigram_dim"].as<unsigned>(),
                              bigram_pretrained ,corpus, 2);
    _INFO <<  bigram_pretrained.size() <<" pretrained bigram embeddings is loaded.";
  }

  if (conf.count("ounigram")) {
    load_pretrained_embedding(conf["ounigram"].as<std::string>(), conf["unigram_dim"].as<unsigned>(), ounigram_pretrained,
                              corpus, 1);
    _INFO << ounigram_pretrained.size() <<" pretrained out-of-domain unigram embedding is loaded.";
  }

  if (conf.count("obigram")) {
    load_pretrained_embedding(conf["obigram"].as<std::string>(), conf["bigram_dim"].as<unsigned>(),
                              obigram_pretrained ,corpus, 2);
    _INFO <<  obigram_pretrained.size() <<" pretrained out-of-domain bigram embeddings is loaded.";
  }
  //create Model
  Model model;

  _INFO << "building ...";
  CRFBuilder engine(&model, conf, corpus, unigram_pretrained, bigram_pretrained,
                       ounigram_pretrained, obigram_pretrained);
  _INFO << "building finishes";

  if (is_train) {
    LOG(INFO) << "there are total " << corpus.n_train << " sents.";
    corpus.load_training_data(conf["dev"].as<std::string>(), "dev");
    LOG(INFO) << "there are total " << corpus.n_dev << " sents.";
    if (conf.count("test")) {
      //corpus.load_test_data(conf["test"].as<std::string>());
      corpus.load_training_data(conf["test"].as<std::string>(), "test");
      LOG(INFO) << "there are total " << corpus.n_test << " sents.";
    }
    std::string outfile = conf["outfile"].as<std::string>();
    train(conf, model, model_name, corpus, training_unigram_vocab,
          training_bigram_vocab, uni_unks, bi_unks, engine, outfile);
    _INFO << "training done";
  }
  else{
    // load model
    _INFO << "loading model";
    std::ifstream in(model_name.c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
    _INFO << "loading model finised";
    corpus.load_test_data(conf["test"].as<std::string>());
    //corpus.load_training_data(conf["test"].as<std::string>(), "test");
    std::string outfile = conf["outfile"].as<std::string>();
    _INFO << "resluts will be written to " << outfile;
    segment(corpus, engine, training_unigram_vocab, training_bigram_vocab, outfile);
  }

  _INFO << "program over";
  LOG(INFO) << "all done";
  return 0;
}
