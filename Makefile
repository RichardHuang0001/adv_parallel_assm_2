# Compiler
NVCC = nvcc
CC = /usr/bin/gcc-11

# Directories
CUDA_PATH = /opt1/cuda/cuda-12.1
SRC = floyd_gpu.cu
TARGET = floyd_gpu

# Compiler flags
NVCC_FLAGS = -ccbin $(CC)
DEBUG_FLAGS = -DDEBUG  # Debugging flag

# Set environment variables
export PATH := $(CUDA_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

# Determine the output target depending on whether DEBUG is set
ifeq ($(DEBUG), 1)
    TARGET := $(TARGET)_db  # Append _db to target name for debug builds
    NVCC_FLAGS += $(DEBUG_FLAGS)
endif

# Build the executable
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

# Clean the build
.PHONY: clean
clean:
	rm -f $(TARGET) $(TARGET)_db
