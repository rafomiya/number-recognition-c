OBJS	= main.o Dataset.o NeuralNetwork.o utils.o
FLAGS	 = -g -c -Wall
LFLAGS	 = -lm

all: $(OBJS)
	gcc -g $(OBJS) -o main $(LFLAGS)

main.o: main.c
	gcc $(FLAGS) main.c -std=c11

Dataset.o: Dataset.c
	gcc $(FLAGS) Dataset.c -std=c11

NeuralNetwork.o: NeuralNetwork.c
	gcc $(FLAGS) NeuralNetwork.c -std=c11

utils.o: utils.c
	gcc $(FLAGS) utils.c -std=c11

clean:
	rm -f $(OBJS) main

run: main
	./main