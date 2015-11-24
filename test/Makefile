IDIR =../
CC=g++
CFLAGS=-I$(IDIR)

simple : simple.cpp
	$(CC) -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f simple
