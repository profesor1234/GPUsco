# Scorbits GPU Miner — Makefile
# Usage: make                  (auto-detect GPU)
#        make SM=86            (RTX 30xx)
#        make SM=89            (RTX 40xx)
#        make SM=61            (GTX 1060)

NVCC    := nvcc
TARGET  := scorbits_gpu
SRC     := scorbits_gpu.cu
LIBS    := -lcurl
CFLAGS  := -O3 -lineinfo

# Auto-detect SM if not specified
ifndef SM
  SM := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  ifeq ($(SM),)
    SM := 86
  endif
endif

ARCH := -arch=sm_$(SM)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(ARCH) -o $(TARGET) $(SRC) $(LIBS)
	@echo ""
	@echo "Built: ./$(TARGET) (sm_$(SM))"
	@echo "Run:   ./$(TARGET) --address SCOyouraddresshere"

# Multi-arch build (RTX 30xx + 40xx)
multi: $(SRC)
	$(NVCC) $(CFLAGS) \
	  -gencode arch=compute_86,code=sm_86 \
	  -gencode arch=compute_89,code=sm_89 \
	  -o $(TARGET) $(SRC) $(LIBS)
	@echo "Built multi-arch: sm_86 + sm_89"

clean:
	rm -f $(TARGET)

.PHONY: all multi clean
