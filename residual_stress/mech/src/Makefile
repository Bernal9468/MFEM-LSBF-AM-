MFEM_DIR = /work/gyj329/MFEM/implementations/bridge_4layer_kokkos/My_kokkos/mfem_openmp/src
include $(MFEM_DIR)/config/config.mk
MKL_INCDIR = /apps/intel/oneapi/2024.2.0.634/2024.2/include
MKL_LIBDIR = /home/gyj329/anaconda3/lib
PARDISO_INC = -I$(MKL_INCDIR)
PARDISO_LIB = -L$(MKL_LIBDIR) -Wl,-rpath,$(MKL_LIBDIR) -lmkl_rt -lpthread -lm -ldl
ENZYME_ROOT = /work/gyj329/MFEM/implementations/bridge_case_enzyme/third_party/enzyme_build
ENZYME_PLUGIN = $(firstword $(wildcard $(ENZYME_ROOT)/Enzyme/ClangEnzyme-*.so))
ENZ_CLANGXX = /usr/lib64/llvm21/bin/clang++
rs_kernels.o: rs_kernels.cpp rs_kernel.hpp rs_kernels.h
	$(ENZ_CLANGXX) -O2 -std=c++17 -fPIC -fno-exceptions -fplugin=$(ENZYME_PLUGIN) -DMFEM_USE_ENZYME -I$(MFEM_DIR) -c $< -o $@
rs_mech: rs_mech.cpp rs_kernels.o tie_constraint.hpp pardiso_direct.hpp rs_kernel.hpp rs_kernels.h $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -std=c++17 -fopenmp $(PARDISO_INC) -I. $< rs_kernels.o -o $@ $(MFEM_LIBS) $(PARDISO_LIB) -fopenmp
