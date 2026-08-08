# LGCR — Luma-Guided Chroma Reconstruction (VapourSynth plugin)
#
# VS headers are expected via VSINCLUDE (override for your install).

VSINCLUDE ?= $(HOME)/vapoursynth/lib/python3.14/site-packages/vapoursynth/include

CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra -mavx2 -mfma
LDFLAGS  ?= -shared

SRCS    := src/maps.cpp src/kernels.cpp src/recon.cpp src/algos.cpp src/plugin.cpp
OBJS    := $(SRCS:.cpp=.o)
HDR     := src/lgcr.h

TARGET  := liblgcr.so

all: $(TARGET)

$(TARGET): $(SRCS) $(HDR)
	$(CXX) $(CXXFLAGS) -I$(VSINCLUDE) $(LDFLAGS) -o $@ $(SRCS)

# Scalar reference build (no AVX2) for correctness cross-checking
liblgcr_scalar.so: $(SRCS) $(HDR)
	$(CXX) -O3 -std=c++17 -fPIC -Wall -Wextra -DLGCR_SUFFIX='"_scalar"' -I$(VSINCLUDE) -shared -o $@ $(SRCS)

clean:
	rm -f $(TARGET) liblgcr_scalar.so src/*.o

.PHONY: all clean
