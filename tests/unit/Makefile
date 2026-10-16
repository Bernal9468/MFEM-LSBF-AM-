# Unit test of the thermal Enzyme kernels (opt/src/fb_enzyme_kernels.*), no MFEM/mesh/solver needed at run time.
#   make test       build against the REAL Enzyme kernels (cluster; same toolchain as opt/src/Makefile) and run
#   make selftest   build against mock_kernels.cpp (FD stand-in) and run -- checks the harness itself, any machine
# Toolchain paths default to the cluster values in opt/src/Makefile; override on the command line, e.g.
#   make test ENZ_CLANGXX=clang++ ENZYME_PLUGIN=/path/ClangEnzyme-21.so MFEM_DIR=/path/mfem
KSRC          ?= ../../opt/src
MFEM_DIR      ?= /work/gyj329/MFEM/implementations/bridge_4layer_kokkos/My_kokkos/mfem_openmp/src
ENZYME_ROOT   ?= /work/gyj329/MFEM/implementations/bridge_case_enzyme/third_party/enzyme_build
ENZYME_PLUGIN ?= $(firstword $(wildcard $(ENZYME_ROOT)/Enzyme/ClangEnzyme-*.so))
ENZ_CLANGXX   ?= /usr/lib64/llvm21/bin/clang++
HOSTCXX       ?= c++
CXXFLAGS_T     = -O2 -std=c++17 -Wall -Wextra -I$(KSRC) -I.

.PHONY: test selftest clean
test: test_fb_kernels
	./test_fb_kernels
selftest: test_fb_kernels_mock
	./test_fb_kernels_mock

# kernels: exactly the flags of opt/src/Makefile's fb_enzyme_kernels.o rule
fb_enzyme_kernels.o: $(KSRC)/fb_enzyme_kernels.cpp $(KSRC)/fb_enzyme_kernels.h $(KSRC)/src_kernel.hpp
	@test -n "$(ENZYME_PLUGIN)" || { echo "ENZYME_PLUGIN not found (set ENZYME_ROOT or ENZYME_PLUGIN)"; exit 1; }
	$(ENZ_CLANGXX) -O2 -std=c++17 -fPIC -fno-exceptions -fplugin=$(ENZYME_PLUGIN) -DMFEM_USE_ENZYME -I$(MFEM_DIR) -c $< -o $@
test_fb_kernels: test_fb_kernels.cpp ref_model.hpp fb_enzyme_kernels.o
	$(ENZ_CLANGXX) $(CXXFLAGS_T) test_fb_kernels.cpp fb_enzyme_kernels.o -o $@ -lm
test_fb_kernels_mock: test_fb_kernels.cpp ref_model.hpp mock_kernels.cpp
	$(HOSTCXX) $(CXXFLAGS_T) test_fb_kernels.cpp mock_kernels.cpp -o $@ -lm
clean:
	rm -f test_fb_kernels test_fb_kernels_mock fb_enzyme_kernels.o
