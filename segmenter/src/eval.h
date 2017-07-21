#ifndef LSTM_CNN_EVAL_H
#define LSTM_CNN_EVAL_H


inline std::tuple<unsigned, unsigned, unsigned> eval_sentence(std::vector<unsigned> &preds, std::vector<unsigned> &golds, std::vector<std::string> &id2labels){
  BOOST_ASSERT_MSG(preds.size() == golds.size(), "preds and golds size not match");
  unsigned n_gold_words = 0;
  unsigned n_pred_words = 0;
  unsigned n_right_words = 0;

  std::vector<std::string> pred_labels;
  std::vector<std::string> gold_labels;
  for (auto &id : preds) { pred_labels.push_back(id2labels[id]); }
  for (auto &id : golds) { gold_labels.push_back(id2labels[id]); }

  for (auto &label: pred_labels) {
    if (label == "E" or label == "S") { n_pred_words += 1; }
  }
  for (auto &label: gold_labels) {
    if (label == "E" or label == "S") { n_gold_words += 1; }
  }

  unsigned start = 0;
  unsigned n = static_cast<int>(preds.size());

  for (unsigned i=0; i<n; i++){
    if (pred_labels[i] == "S" and gold_labels[i] == "S")
      n_right_words += 1;
    else if (gold_labels[i] == "B")
      start = i;
    else if (gold_labels[i] == "E"){
      bool flag = true;
      for (unsigned j=start; j<=i; j++){
        if (pred_labels[j] != gold_labels[j]){
          flag = false;
          break;
        }
      }
      if (flag) { n_right_words += 1; }
    }
  }

  return std::make_tuple(n_gold_words, n_pred_words, n_right_words);
};


#endif //LSTM_CNN_EVAL_H
