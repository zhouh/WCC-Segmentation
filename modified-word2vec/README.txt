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
