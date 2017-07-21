#!/bin/bash
TRAINDATA=../data/corpus.train                  #Use text data from <file> to train the model
OUTPUTEMBEDDING=../result/embedding-sg-neg-100.train       #Use <file> to save the resulting word vectors / word-clusters
WORDDIM=100                                    #Set size of word vectors
WINDOWSIZE=5                                   #Set max skip length between words
DOWNSAMPLEDTHRES=1e-3                          #Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled. useful range is (0, 1e-5)
ISHS=0                                         #Whethter use hierarchical softmax
NEGNUM=5                                       #Number of negative examples, common values are 3-10 (0 = not used)
THREADS=18                                     #Use <int> threads
ITERNUM=5                                      #Run <int> training iterations
MINCOUNT=5                                     #This will discard words that appear less than <int> times
ALPHA=0.025                                    #Set starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
CLASSES=0                                      #Output word classes rather than word vectors; (0 = vectors are written)
DEBUG=2                                        #Set the debug mode (2 = more info during training)
BINARY=0                                       #Save the resulting vectors in binary model; (0 = off)
SAVEVOCAB=../result/vocabulary.test            #The vocabulary will be saved to <file>
READVOCAB=""                                   #The vocabulary will be read from <file>, not constructed from the training data
CBOW=0                                         #Use the continuous bag of words model; (0 for skip-gram, 1 for CBOW)

#valgrind --track-origins=no 
./word2vec -train ${TRAINDATA} -output ${OUTPUTEMBEDDING} -save-vocab ${SAVEVOCAB} -binary ${BINARY} \
    -alpha ${ALPHA} -cbow ${CBOW} \
    -size ${WORDDIM} -sampe ${DOWNSAMPLEDTHRES} -negative ${NEGNUM} -min-count ${MINCOUNT} -window ${WINDOWSIZE} \
    -hs ${ISHS} -classes ${CLASSES} \
    -threads ${THREADS} -iter ${ITERNUM} \
    -debug ${DEBUG}
