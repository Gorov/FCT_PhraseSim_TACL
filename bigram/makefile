CC = g++
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result

OBJS = BinaryLearnerNew.o ClasificationLearnerNew.o FeatureModel.o MultiBinaryLearner.o SkipgramLearner.o BaseLearner.o BinaryThresLearner.o EmbeddingModel.o JointLearner.o GetPhraseSim.o PPDBLearnerB2U.o MTLTrainer.o

all: TrainPPDB 

%.o : %.cpp
	$(CC) -c $< -o $@ $(CFLAGS)

PhraseEmb : PhraseEmb.cpp $(OBJS)
	$(CC) PhraseEmb.cpp $(OBJS) -o PhraseEmb $(CFLAGS)

TrainPPDB : TrainPPDB.cpp $(OBJS)
	$(CC) TrainPPDB.cpp $(OBJS) -o TrainPPDB $(CFLAGS)

PredictPPDB : PredictPPDB.cpp $(OBJS)
	$(CC) PredictPPDB.cpp $(OBJS) -o PredictPPDB $(CFLAGS)

PredictSim : PredictSim.cpp $(OBJS)
	$(CC) PredictSim.cpp $(OBJS) -o PredictSim $(CFLAGS)

PredictML : predictML.cpp $(OBJS)
	$(CC) predictML.cpp $(OBJS) -o PredictML $(CFLAGS)

run: run.cpp $(OBJS)
	$(CC) run.cpp $(OBJS) -o run $(CFLAGS)

test: test.cpp $(OBJS)
	$(CC) test.cpp $(OBJS) -o test $(CFLAGS)

train: train.cpp $(OBJS)
	$(CC) train.cpp $(OBJS) -o train $(CFLAGS)

train_lm_nce: train_lm_nce.cpp $(OBJS)
	$(CC) train_lm_nce.cpp $(OBJS) -o train_lm_nce $(CFLAGS)

clean:
	rm -rf TrainPPDB PredictPPDB PredictSim PhraseEmb run test *.o
