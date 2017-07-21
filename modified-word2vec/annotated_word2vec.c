#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;

struct vocab_word {
  long long cn; // 词频,来自于vocab file或者从训练模型中来计算
  int *point;   // 霍夫曼树中从根节点到该词的路径，存放路径上每个非叶结点的索引
  char *word, *code, codelen; //word：string 字面值 code：Huffman编码 codelen：huffman编码的长度
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
//输入文件中每个基本词的结构体数组
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2;
int window = 5, min_count = 5;//在由语料库构建词典(vocab数组)时，剔除词频小于min_count的词。
//构建词典后仍需要判断是否需要对低频次进行清理，
//如果词典的大小N>0.7*vocab_hash_size,则从词典中删除所有词频小于min_reduce的词。
int num_threads = 12, min_reduce = 1;
//该数组存文件中基本词的字面的hash码，和基本词在vocab_word数组中的位置
//其中基本词的字面的hash码作为该数组的下标
int *vocab_hash;
//vocab_size 不同单词的个数，也就是词典的大小
//layer1_size 词向量的长度
//file_size训练文件的大小
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
//syn0 － 存储词典中每个词的词向量
//syn1 － 存储Huffman树各个内节点对应的向量
//syn1neg －负采样时，存储每个词对应的辅助向量
//expTable: sigmoid函数表，提前计算好，提高效率
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

//每个单词的能量分布表，table在负采样中用到
void InitUnigramTable() {

  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));//分配空间
  //遍历词汇表，统计词的能量总值train_words_pow
  for (a = 0; a < vocab_size; a++)
    train_words_pow += pow(vocab[a].cn, power);


  i = 0;
  //表示已遍历词的能量值占总能量的比
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  //a - table表的索引
  //i - 词汇表的索引
  for (a = 0; a < table_size; a++) {
    table[a] = i;//单词i占用table的a位置
                 //table反映的是一个单词能量的分布，一个单词能量越大，所占用的table的位置越多
    if (a / (double)table_size > d1) {
      i++;       //移到下个词
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    // put everthing else in the end of the unigram table??？
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space(' ') + tab(\t) + EOL(\n) to be word boundaries
//从fin中读一个词到字符串word
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;//a - 用于向word中插入字符的索引；ch - 从fin中读取的每个字符
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;//回车，开始新的一行，重新开始while循环读取下一个字符
    //当遇到space(' ') + tab(\t) + EOL(\n)时，认为word结束
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {//跳出while循环，这里的特例是‘\n’，我们需要将回退给fin，词汇表中'\n'用</s>来表示。
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {//此时word还为空(a=0)，直接将</s>赋给word
        strcpy(word, (char *)"</s>");
        return;
      }
      else continue;//此时a＝0，且遇到的为\t or ' '，直接跳过取得下一个字符
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;//字符串末尾以/0作为结束符
}

//Returns hash value of a word
//返回一个词的hash值，一个词跟hash值一一对应（可能冲突）
//hash规则
//hash = hash * 257 + word[a];
//@param word 基本词字面
//@return 每个基本词的hash编码
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];//int和char相加，相当于加了char的ansii值
  hash = hash % vocab_hash_size;
  return hash;
}

// 返回一个词在词汇表中的位置，如果不存在则返回-1
// @brief 根据word字面的hash查找其在vocab中的位置,即vocab_hash[hash]的值
// @param word 字面
// @return vocab_hash[hash]的值
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
// 从文件流中读取一个词，并返回这个词在词汇表中的位置
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;//当文件只有一个EOF字符时，当将EOF读入word后，_IOEOF被设置，达到文件尾。
  return SearchVocab(word);
}

// Adds a word to the vocabulary 将一个词添加到一个词汇中，返回该词在词汇表中的位置
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));//在当前词汇表末尾添加word
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;//词频记为0
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  //由于hash表初始化为－1
  while (vocab_hash[hash] != -1)
   hash = (hash + 1) % vocab_hash_size;//使用开放地址法解决冲突
  vocab_hash[hash] = vocab_size - 1;//由词的hash值找到她所在词汇表的排序位置
  return vocab_size - 1;
}

// Used later for sorting by word counts
//根据词频从大到小排序
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
// 根据词频排序
// 通过排序把出现数量少的word排在vocab数组的后面
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position？？？ 这是因为:
  // 在LearnVocabFromTrainFile将</s>放在vocab的第一个位置
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {//不考虑</s>小于min_count的情况
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;//处理冲突
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  //如果词典的大小N>0.7*vocab_hash_size,则从词典中删除所有词频小于min_reduce的词。
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);

  vocab_size = b;
  // reset the hash table
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i;
  long long min1i, min2i;// two smallest nodes
  long long pos1, pos2; // current pivots
  long long point[MAX_CODE_LENGTH]; //记录从root到word的路径
  char code[MAX_CODE_LENGTH];
  // calloc initializes the memory to zeros
  // SHOULD IT BE vocab_size * 2 - 1 - this is because
  // it seems that </s> is in part of construction, but vocab_size is
  // actually len(vocab)-1, excluding </s>
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  // count - word counts of all words
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  // extend count as twice large
  // SO ONLY vocab_size * 2 - 1 elements will BE NEEDED, EVEN FOR A
  // COMPLETE TREE
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  // initialize the node positions
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  // the vocab should have been sorted IN DECREASING order
  // pos1 and pos2 will be the two current smallest node
  // they could be the original elements, they could be composed parent nodes
  // the parents node will be constructed and placed along the array
  // from [vocab_size to vocab_size * 2 - 1].
  // THE GOOD THING IS: the elements on the left of pos1 are all SORTED,
  // and the elements on the right of pos2 will also be SORTED.

  // THE LAST WORD </s> WILL ALSO BE INCLUDED IN THE TREE
  // ONLY NEED TO CONSTRUCT vocab_size - 1 times, that is max number
  // of parent nodes for a complete binary tree
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1i, min2i'
    // MIN1 goes first
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {// move right via pos2
        min1i = pos2;
        pos2++;
      }
    } else {// no choice, can move right ONLy now
      min1i = pos2;
      pos2++;
    }
    // MIN2 goes next
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {// move left via pos1
        min2i = pos1;
        pos1--;
      } else {// move right via pos2
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    // parent's count is the sum of children's counts
    count[vocab_size + a] = count[min1i] + count[min2i];
    // commmon parents
    // level 2 parents will be from vocab_size to vocab_size * 2
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    // binary code: min1i 0 min2i 1, for each leaf and internal node
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  // update each vocab word and its parent
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;// the depth of traverse from leaf to root
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;// depth
    // point - relative index of parent from vocab_size+1
    // in reverse order - path from root to the current word (leaf node)
    vocab[a].point[0] = vocab_size - 2;
    //下面存放每个基本词的路径，注意i - b - 1是距离叶子节点最近的父节点的编码
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;// parent node index
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;

  // THis is consistent with ReadVocab()？？？
  AddWordToVocab((char *)"</s>");//最初将</s>添加到vocab的第一个位置
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;

    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }

    i = SearchVocab(word);//SearchVocab返回word在vocab中的位置，如果不存在返回－1.
    // no found in vocab - add to vocab and vocab_hash, set word.cn = 1
    // found in vocab - update word.cn += 1
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    // vocab is too LARGE for the current vocab_hash_table
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    }//end of while:vocab 和 vocab_hash 建立完成
  SortVocab();//在该方法中将词频<min_count的词都舍去
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}
// 输出单词和词频到文件
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  // "</s>" will be the first line of the written voc file
  for (i = 0; i < vocab_size; i++)
    fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;// i －number of words in vocab
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  //给Hash表初始化
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  // read all the words, and their counts from the file
  // add them in the vocab, add their index to vocab_hash
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);//AddWordToVocab返回word在vocab的位置
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);//这里的c是什么？？？
    i++;
  }
  SortVocab();//排序中删除词频小于min_count的词
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");//读train_file主要是为了确定训练文件的大小，
                                //在LearnVocabFromTrainFile中最后也进行了此步骤。
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}
//初始化神经网络结构
void InitNet() {
  long long a, b;
  // layer1_size will the the dimension of feature space
  // syn0 and syn1/syn1neg are of size vocab_size * layer1_size
  // syn0-词向量
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  // Hierarchical Softmax
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++)//将Huffman树的内节点初始化为0
     for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  // Negative Sampling
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++)//将每个词的辅助向量初始化为0
     for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) //初始化词向量，每一维都是[-0.5,0.5]/layer1_size的随机数
    for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree();
}
//这个线程函数执行之前，已经做好了一些工作：根据词频排序的词汇表，每个单词的huffman编码
void *TrainModelThread(void *id) {
  long long a, b, d, cw, word;//cw-｜context(w)｜
  // word 向sen中添加单词用，句子完成后表示句子中的当前单词
  // last_word 上一个单词，辅助扫描窗口
  // sentence_length 当前句子的长度（单词数）
  // sentence_position 当前单词在当前句子中的index
  long long last_word, sentence_length = 0, sentence_position = 0;
  // word_count 已训练语料总长度
  // last_word_count 保存值，以便在新训练语料长度超过某个值时输出信息
  long long word_count = 0, last_word_count = 0;
  //此处注意sen数组存放的是当前训练样本，即基本词的上下文在vocab中的位置
  long long sen[MAX_SENTENCE_LENGTH + 1];
  // l1 ns中表示word在concatenated word vectors中的起始位置，之后layer1_size是对应的word vector，因为把矩阵拉成长向量了
  // l2 cbow或ns中权重向量的起始位置，之后layer1_size是对应的syn1或syn1neg，因为把矩阵拉成长向量了
  // c 循环中的计数作用
  // target ns中当前的sample
  // label ns中当前sample的label
  long long l1, l2, c, target, label, local_iter = iter;
  // id 线程创建的时候传入，辅助随机数生成
  unsigned long long next_random = (long long)id;
  // f e^x / (1/e^x)，fs中指当前编码为是0（父亲的左子节点为0，右为1）的概率，ns中指label是1的概率
  // g 误差(f与真实值的偏离)与学习速率的乘积
  real f, g;
  // 当前时间，和start比较计算算法效率
  clock_t now;
  // neu1对应于Xw，即如果是CBOW，neu1是context(w)各vector的累加和；如果是skip－gram的话，neu1是w对应的vector
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  // 误差累计项
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
   //每个线程对应一段文本。根据线程id找到自己负责的文本的初始位置
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {

    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        //打印训练进度情况 回车的ascii码是13 alpha是训练速率
        printf("%cAlpha: %f Progress: %.2f%% Words/thread/sec: %.2fk ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      //自适应调整学习率
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }


    if (sentence_length == 0) {
      while (1) {
        //从文件流中读取一个词，并返回这个词在词汇表中的位置
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;//是个'\n'，句子结束
        //对高频词进行亚采样，亚采样可以提高2～10倍的训练速度，并且可以使低频词向量更精确，具体来说：
        //以1-ran的概率舍弃高频词
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        //sen存放的为该词在词典中的索引,并且sen[]中词的顺序与文本中词的顺序一致。
        sen[sentence_length] = word;
        sentence_length++;
        //1000个单词视作一个句子
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      //充当最外面while循环的指针，依次扫描句中每个word
      sentence_position = 0;
    }


    //如果当前线程已处理的单词超过了阈值，要么进行新一轮的迭代，要么在迭代iter次后退出
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    //取出当前单词
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    //［0，window－1］的随机数，用于确定|context(w)|
    b = next_random % window;



    //～～～～～～～～～～～～～～～～～～～～～train the cbow architecture (context(w),w)～～～～～～～～～～～～～～～～～～～～
    if (cbow) {

      cw = 0;//|context(w)|，初始化为0
      for (a = b; a < window * 2 + 1 - b; a++) //求向量和Xw
        if (a != window) {//当a＝window时，c=sentence_position,当前词上下文中不应该包括
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];//c处存放的单词在词典中的索引
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) //step1:传说中的向量和，layer1_size是词向量的维度，默认是100
        neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }

      if (cw) {
        for (c = 0; c < layer1_size; c++) //求平均向量和
          neu1[c] /= cw;

        //层次softmax优化
        if (hs)
          for (d = 0; d < vocab[word].codelen; d++) {//遍历Huffman树上的内节点
          f = 0;
          //l2为内节点在syn1的起始位置
          l2 = vocab[word].point[d] * layer1_size;

          for (c = 0; c < layer1_size; c++)//step2:f为Xw和Theta的内积
            f += neu1[c] * syn1[c + l2];

          if (f <= -MAX_EXP) continue;//sigmoid(f)=0
          else if (f >= MAX_EXP) continue;//sigmoid(f)=1
          //step3:对f进行sigmoid变换
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          g = (1 - vocab[word].code[d] - f) * alpha;// step4:alpha*[1-dj-sigmoid(f)]
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];//step5:计算累计误差，用于更新词向量
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];//update1:更新内节点Theta
        }

        // 负采样优化
        if (negative > 0)
          for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;//目标单词
            label = 1;//正样本
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;//负采样抽到回车，再重新采样
            if (target == word) continue;
            label = 0;//负样本
          }

          l2 = target * layer1_size;//l2-target对应辅助变量Theta(syn1neg)的起始位置
          f = 0;

          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];//在负采样优化中，每个word都对应一个辅助向量Theta(syn1neg)
          if (f > MAX_EXP) g = (label - 1) * alpha;//sigmoid(f)=1
          else if (f < -MAX_EXP) g = (label - 0) * alpha;//sigmoid(f)=0
          //g=alpha*[label-sigmoid(Xw*Theta)]
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          //neu1e-累积误差，直到一轮抽样完了后才能更新输入层的词向量。
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];//隐藏层的误差
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];//update1:更新辅助变量Theta(syn1neg)
        }

        //对context(w)中的每个词向量进行更新
        for (a = b; a < window * 2 + 1 - b; a++)
          if (a != window) {//cbow模型 更新的不是中间词语的向量，而是周围几个词语的向量。
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++)
            syn0[c + last_word * layer1_size] += neu1e[c];//update2:更新context(w)中的每一个词向量
        }
      }
    }//end of cbow

    // ～～～～～～～～～～～～～～～～～～～～train the skip-gram architecture (w,context(w))～～～～～～～～～～～～～～～～～～～
    else {
      for (a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];//窗口中待处理的词(公式推导中的u)
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;//l1-u在syn0层的起始位置
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;//累计误差初始化


        // 层次Softmax，这里用到了模型对称：p(u|w)=p(w|u),其中u是context(w)中每个词
        if (hs)
         for (d = 0; d < vocab[word].codelen; d++) {//遍历word对应的内节点
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;//l2-内节点在syn1层的位置
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];//V_u和Theta_w的内积
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];//sigmoid(f）
          //g=alpha*[1-d_j-sigmoid(V_u和Theta_w的内积)]
          g = (1 - vocab[word].code[d] - f) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];//计算累计误差，用于更新V_u
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];//update1:更新内节点
        }


        // 负采样 (w,context(w))
        if (negative > 0)

          for (d = 0; d < negative + 1; d++) {

          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }

          l2 = target * layer1_size;//l2-负采样的词在syn1neg的初始位置 l1-context(w)中的每个词在syn0层的初始位置
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];//context(w)中的词与负采样的词做内积
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          //g=alpga*[label-sigmoid(context(w)中的词与负采样的词做内积)]
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];//计算累计误差
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];//update1:更新负采样的词对应的辅助向量
        }

        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];//update2:更新context(w)中的每个词向量
      }
    }//end of skip gram
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }//end of while(1)
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  // 创建多线程
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  //先从词汇表中加载，否则从训练文件中加载
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  //输出词汇表文件，词＋词频
  if (save_vocab_file[0] != 0) SaveVocab(); //根据需要来判断是否进行vocab(排序好的)的保存
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++)
    pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++)
    pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary)
        for (b = 0; b < layer1_size; b++)
          fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else
        for (b = 0; b < layer1_size; b++)
          fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes;//cluser count
    int iter = 10;
    int closeid;
    //属于每个中心的单词个数
    int *centcn = (int *)malloc(classes * sizeof(int));
    // 存放每个单词指派的中心id
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    //存放每个中心的向量表示
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    //随机指派每个单词的中心
    for (a = 0; a < vocab_size; a++)
      cl[a] = a % clcn;

    for (a = 0; a < iter; a++) { //一共迭代iter轮
      //中心向量坐标初始化为0
      for (b = 0; b < clcn * layer1_size; b++)
        cent[b] = 0;
      //属于每个中心的单词数为1
      for (b = 0; b < clcn; b++)
        centcn[b] = 1;
      //将属于每个中心的点的每个坐标相加
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++)
          cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;// 分别计算属于每个中心的点个数
      }

      for (b = 0; b < clcn; b++) {//更新每个中心的向量表示？？
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];//每个中心的平均坐标
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];//每个中心的每维的平方和
        }
        closev = sqrt(closev);//每个中心的向量模
        for (c = 0; c < layer1_size; c++) //将中心向量归一化
          cent[layer1_size * b + c] /= closev;
      }

      for (c = 0; c < vocab_size; c++) {//更新每个样本的中心，循环迭代每个词
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {//判断该词该归为哪一类
          x = 0;
          for (b = 0; b < layer1_size; b++) //归一化聚类中心和词向量做内积
            x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {// 词向量也归一化的，那么两个点的距离最小等价于内积最大，可是
                           // 词向量并没有归一化啊？？？
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
          printf("Argument missing for %s\n", str);
          exit(1);
        }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // e()
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // sigmoid()
  }
  TrainModel();
  return 0;
}
