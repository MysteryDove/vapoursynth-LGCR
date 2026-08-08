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
PYTHON  ?= $(HOME)/vapoursynth/bin/python3
ASAN_FLAGS := -O1 -g -std=c++17 -fPIC -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer

all: $(TARGET)

$(TARGET): $(SRCS) $(HDR)
	$(CXX) $(CXXFLAGS) -I$(VSINCLUDE) $(LDFLAGS) -o $@ $(SRCS)

# Scalar reference build (no AVX2) for correctness cross-checking
liblgcr_scalar.so: $(SRCS) $(HDR)
	$(CXX) -O3 -std=c++17 -fPIC -Wall -Wextra -DLGCR_SUFFIX='"_scalar"' -I$(VSINCLUDE) -shared -o $@ $(SRCS)

liblgcr_asan.so: $(SRCS) $(HDR)
	$(CXX) $(ASAN_FLAGS) -I$(VSINCLUDE) -shared -o $@ $(SRCS)

check: $(TARGET) liblgcr_scalar.so
	$(PYTHON) test/test_lgcr.py
	$(PYTHON) test/test_algo6.py
	$(PYTHON) test/test_regressions.py
	$(PYTHON) test/battery.py --all --check
	$(PYTHON) -m evaluation.test_protocol
	$(PYTHON) -m evaluation.check_paper

paper-check:
	$(PYTHON) -m evaluation.check_paper

eval-dev: $(TARGET)
	$(PYTHON) -m evaluation.run --tune --write-results

eval-test: $(TARGET)
	$(PYTHON) -m evaluation.run --split test --write-results

eval-ablation: $(TARGET)
	$(PYTHON) -m evaluation.run --split test --ablation --write-results

eval-siting: $(TARGET)
	$(PYTHON) -m evaluation.run --split test --siting-mismatch --write-results

eval-phase: $(TARGET)
	$(PYTHON) -m evaluation.phase_rescue --write-results

eval-kernels: $(TARGET)
	$(PYTHON) -m evaluation.kernel_study --write-results

eval-kernel-confirm: $(TARGET)
	$(PYTHON) -m evaluation.kernel_confirm --write-results

eval-kernel-study: eval-kernels eval-kernel-confirm

eval-wada-confirm: $(TARGET)
	$(PYTHON) -m evaluation.wada_confirm --write-results

eval-corpora:
	$(PYTHON) -m evaluation.coedge --write-results

eval-results: eval-dev eval-test eval-ablation eval-siting eval-phase

asan-check: liblgcr_asan.so
	ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0 \
	LGCR_PLUGIN="$(CURDIR)/liblgcr_asan.so" $(PYTHON) test/test_regressions.py --asan

clean:
	rm -f $(TARGET) liblgcr_scalar.so liblgcr_asan.so src/*.o

.PHONY: all check paper-check asan-check eval-dev eval-test eval-ablation eval-siting eval-phase eval-kernels eval-kernel-confirm eval-kernel-study eval-wada-confirm eval-corpora eval-results clean
