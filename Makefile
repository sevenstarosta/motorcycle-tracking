CPP=g++
OFILES= crosswalk.o
LIBS= -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_objdetect -lopencv_video

%.o : %.cpp
	$(CPP) -O2 -c -o $@ $<

all: velocidade old

velocidade: velocidade.o
	$(CPP) -o $@ $^ $(LIBS)

old: subtraction.o
	$(CPP) -o $@ $^ $(LIBS)
