PROGRAMMING_EXAMPLES_TESTDIR = $(OUT)/programming_examples
PROGRAMMING_EXAMPLES_OBJDIR = $(OBJDIR)/programming_examples

PROGRAMMING_EXAMPLES_INCLUDES = $(COMMON_INCLUDES)
PROGRAMMING_EXAMPLES_LDFLAGS = -ltt_metal_detail -ltt_metal -lllrt -ldevice -lbuild_kernels_for_riscv -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp -ltracy

include $(TT_METAL_HOME)/tt_metal/programming_examples/loopback/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/eltwise_binary/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/profiler/module.mk


PROFILER_TESTS += \
		  programming_examples/profiler/test_custom_cycle_count\
		  programming_examples/profiler/test_full_buffer


programming_examples: programming_examples/loopback \
                      programming_examples/eltwise_binary \
		      $(PROFILER_TESTS)

programming_examples/loopback: $(PROGRAMMING_EXAMPLES_TESTDIR)/loopback;
programming_examples/eltwise_binary: $(PROGRAMMING_EXAMPLES_TESTDIR)/eltwise_binary;
programming_examples/profiler/%: $(PROGRAMMING_EXAMPLES_TESTDIR)/profiler/%;
