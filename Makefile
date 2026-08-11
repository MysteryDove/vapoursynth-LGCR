# LGCR — Luma-Guided Chroma Reconstruction (VapourSynth plugin)
#
# VS headers are expected via VSINCLUDE (override for your install).

VSINCLUDE ?= $(HOME)/vapoursynth/lib/python3.14/site-packages/vapoursynth/include

CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra -mavx2 -mfma
LDFLAGS  ?= -shared
LDLIBS   ?=

LGCR_ENABLE_CUDA ?= 0
CUDA_ROOT ?= /usr/local/cuda
LGCR_VERSION_MAJOR ?= 2
LGCR_VERSION_MINOR ?= 1

CPPFLAGS += -DLGCR_VERSION_MAJOR=$(LGCR_VERSION_MAJOR) \
	-DLGCR_VERSION_MINOR=$(LGCR_VERSION_MINOR)

ifeq ($(LGCR_ENABLE_CUDA),1)
CPPFLAGS += -DLGCR_ENABLE_CUDA=1 -I$(CUDA_ROOT)/include
LDLIBS += -L$(CUDA_ROOT)/lib64 -Wl,-rpath,$(CUDA_ROOT)/lib64 -lcudart
else
CPPFLAGS += -DLGCR_ENABLE_CUDA=0
endif

SRCS    := src/pipeline.cpp src/cuda_backend.cpp src/workspace.cpp src/maps.cpp src/kernels.cpp src/recon.cpp src/algos.cpp src/bm.cpp src/downsample.cpp src/plugin.cpp
OBJS    := $(SRCS:.cpp=.o)
HDR     := $(wildcard src/*.h)

TARGET  := liblgcr.so
PYTHON  ?= $(HOME)/vapoursynth/bin/python3
BENCHMARK_ARGS ?=
ASAN_FLAGS := -O1 -g -std=c++17 -fPIC -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer
BUILD_CONFIG := .lgcr-build-config

all: $(TARGET)

$(BUILD_CONFIG): FORCE
	@value='cuda=$(LGCR_ENABLE_CUDA);cuda_root=$(CUDA_ROOT);version=$(LGCR_VERSION_MAJOR).$(LGCR_VERSION_MINOR);cxx=$(CXX);cxxflags=$(CXXFLAGS)'; \
	current="$$(test -f $@ && sed -n '1p' $@)"; \
	if test "$$current" != "$$value"; then printf '%s\n' "$$value" > $@; fi

$(TARGET): $(SRCS) $(HDR) $(BUILD_CONFIG)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(VSINCLUDE) $(LDFLAGS) -o $@ $(SRCS) $(LDLIBS)

# Scalar reference build (no AVX2) for correctness cross-checking
liblgcr_scalar.so: $(SRCS) $(HDR) $(BUILD_CONFIG)
	$(CXX) $(CPPFLAGS) -O3 -std=c++17 -fPIC -Wall -Wextra -DLGCR_SUFFIX='"_scalar"' -I$(VSINCLUDE) -shared -o $@ $(SRCS) $(LDLIBS)

liblgcr_asan.so: $(SRCS) $(HDR) $(BUILD_CONFIG)
	$(CXX) $(CPPFLAGS) $(ASAN_FLAGS) -I$(VSINCLUDE) -shared -o $@ $(SRCS) $(LDLIBS)

test/test_pipeline: test/test_pipeline.cpp src/pipeline.cpp src/cuda_backend.cpp src/pipeline.h src/cuda_backend.h
	$(CXX) -O2 -std=c++17 -Wall -Wextra -DLGCR_ENABLE_CUDA=0 -Isrc -o $@ test/test_pipeline.cpp src/pipeline.cpp src/cuda_backend.cpp

test/stream: test/stream.cpp
	$(CXX) -O3 -std=c++17 -Wall -Wextra -pthread -o $@ $<

test/test_sparse_workset: test/test_sparse_workset.cpp src/maps.cpp src/kernels.cpp src/workspace.cpp src/lgcr.h src/pipeline.h src/backend.h
	$(CXX) $(CPPFLAGS) -O2 -std=c++17 -Wall -Wextra -mavx2 -mfma \
		-I$(VSINCLUDE) -Isrc -o $@ test/test_sparse_workset.cpp src/maps.cpp src/kernels.cpp src/workspace.cpp

test/test_plane: test/test_plane.cpp src/lgcr.h src/pipeline.h src/backend.h
	$(CXX) $(CPPFLAGS) -O2 -std=c++17 -Wall -Wextra \
		-I$(VSINCLUDE) -Isrc -o $@ test/test_plane.cpp

plane-check: test/test_plane
	./test/test_plane

sparse-workset-check: test/test_sparse_workset
	./test/test_sparse_workset

numerical-equivalence-check: $(TARGET) liblgcr_baseline.so
	$(PYTHON) test/test_numerical_equivalence.py

plane-sharing-check: $(TARGET)
	$(PYTHON) test/test_plane_sharing.py

stream: test/stream
	./test/stream --threads 1 --cpu-list 0

liblgcr_baseline.so: test/build_baseline.sh $(BUILD_CONFIG)
	VSINCLUDE=$(VSINCLUDE) CXX=$(CXX) CXXFLAGS='$(CXXFLAGS)' \
		bash test/build_baseline.sh $@ $(BASELINE_REF)

baseline-plugin: liblgcr_baseline.so

pipeline-check: test/test_pipeline
	./test/test_pipeline

cuda-framework-check:
	$(CXX) -O2 -std=c++17 -Wall -Wextra -DLGCR_ENABLE_CUDA=1 -Isrc \
		-I$(CUDA_ROOT)/include -o test/test_pipeline_cuda test/test_pipeline.cpp \
		src/pipeline.cpp src/cuda_backend.cpp -L$(CUDA_ROOT)/lib64 \
		-Wl,-rpath,$(CUDA_ROOT)/lib64 -lcudart
	./test/test_pipeline_cuda

check: $(TARGET) liblgcr_scalar.so pipeline-check plane-check plane-sharing-check sparse-workset-check numerical-equivalence-check
	$(PYTHON) test/test_lgcr.py
	$(PYTHON) test/test_algo6.py
	$(PYTHON) test/test_regressions.py
	$(PYTHON) test/test_bm.py
	$(PYTHON) test/test_downsample.py
	$(PYTHON) test/test_backend_matrix.py
	$(PYTHON) test/test_concurrency.py
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

eval-bm: $(TARGET)
	$(PYTHON) -m evaluation.bm_study --write-results

eval-results: eval-dev eval-test eval-ablation eval-siting eval-phase

benchmark: $(TARGET)
	$(PYTHON) test/benchmark.py $(BENCHMARK_ARGS)

uprofile: $(TARGET) test/stream
	$(PYTHON) test/uprofile_matrix.py $(UPROFILE_ARGS)

asan-check: liblgcr_asan.so
	ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0 \
	LGCR_PLUGIN="$(CURDIR)/liblgcr_asan.so" $(PYTHON) test/test_regressions.py --asan
	ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0 \
	LGCR_PLUGIN="$(CURDIR)/liblgcr_asan.so" $(PYTHON) test/test_bm.py
	ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0 \
	LGCR_PLUGIN="$(CURDIR)/liblgcr_asan.so" $(PYTHON) test/test_downsample.py
	ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0 \
	LGCR_PLUGIN="$(CURDIR)/liblgcr_asan.so" $(PYTHON) test/test_concurrency.py

clean:
	rm -f $(TARGET) liblgcr_scalar.so liblgcr_baseline.so liblgcr_asan.so test/test_pipeline \
		test/test_pipeline_cuda test/test_plane test/test_sparse_workset test/stream $(BUILD_CONFIG) src/*.o

.PHONY: all check pipeline-check plane-check plane-sharing-check sparse-workset-check numerical-equivalence-check cuda-framework-check paper-check asan-check benchmark uprofile stream baseline-plugin eval-dev eval-test eval-ablation eval-siting eval-phase eval-kernels eval-kernel-confirm eval-kernel-study eval-wada-confirm eval-corpora eval-bm eval-results clean FORCE

FORCE:
