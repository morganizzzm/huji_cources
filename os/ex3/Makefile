CC=g++
CCFLAGS = -c -Wall -std=c++11 -pthread -g $(INCS)
SOURCES=MapReduceFramework.cpp Barrier.cpp
OBJECTS=$(SOURCES:.cpp=.o)
LIB=libMapReduceFramework.a

all: $(LIB)

$(LIB): $(OBJECTS)
	ar rcs $@ $^

%.o: %.cpp
	$(CC) $(CCFLAGS) $< -o $@

clean:
	rm -f $(LIB) $(OBJECTS)