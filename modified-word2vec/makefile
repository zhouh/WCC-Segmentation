CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -O3 -lm -pthread -march=native -Wall -funroll-loops -Wno-unused-result -std=c11

all: word2vec oseg2vec

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
oseg2vec : scs.c
	$(CC) seg2vec.c -o oseg2vec $(CFLAGS)
#word2phrase : word2phrase.c
#	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
#distance : distance.c
#	$(CC) distance.c -o distance $(CFLAGS)
#word-analogy : word-analogy.c
#	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
#compute-accuracy : compute-accuracy.c
#	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
#	chmod +x *.sh

clean:
	rm -rf word2vec oseg2vec 
