IDIR =../
CC=g++
CFLAGS=-I$(IDIR) -O3

all: simple simple_block

simple : simple.cpp
	$(CC) -o $@ $< $(CFLAGS)

simple_block : simple_block.cpp
	$(CC) -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f simple simple_block
