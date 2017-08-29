#WCC-Segmentation

Segmenter code is in "segmenter" directory.
Embedding training code is in "modified-word2vec" dircectory.

#Run
make

sample training data is in data directory

#Training command

./seg2vec -train data/small.seg.char.giga \
    -output  vecs/example.emb \
    -cbow 0  \
    -size 50 \
    -window 5 \
    -negative 10 \
    -sample 1e-4 \
    -threads 6 \
    -binary 0 \
    -iter 8 \


#Requirements
boost and glog

#RUN
1. ./configure
2. make
3. run scripts are in exp directory

#NOTES
The performance between crf and greedy model is comparable.
You can replace "cmd=greedy" to "cmd=crf" in the scripts.