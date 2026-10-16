# Optimised full-bridge OTI driver (v5 = v3 + mixed solver: CG+BoomerAMG for large Newton systems, PARDISO for sensitivities). Two binaries from one source:
#   make order1   -> fbko_opt_order1 : -dorder 0 / 1 only  (OTI_ORDER=1, ~10x fewer OTI coefficients; the fast production build)
#   make order2   -> fbko_opt_order2 : -dorder 0 / 1 / 2   (OTI_ORDER=2, needed for second-order (A3) sensitivities)
# Same MFEM (legacy-OpenMP, THREAD_SAFE) + MKL PARDISO toolchain as ../src/Makefile.tier1_oti.
MFEM_DIR = /work/gyj329/MFEM/implementations/bridge_4layer_kokkos/My_kokkos/mfem_openmp/src
include $(MFEM_DIR)/config/config.mk

MKL_INCDIR = /apps/intel/oneapi/2024.2.0.634/2024.2/include
MKL_LIBDIR = /home/gyj329/anaconda3/lib
PARDISO_INC = -I$(MKL_INCDIR)
PARDISO_LIB = -L$(MKL_LIBDIR) -Wl,-rpath,$(MKL_LIBDIR) -lmkl_rt -lpthread -lm -ldl
HYPRE_DIR = /work/gyj329/MFEM/hypre_seq_omp/src/hypre
HYPRE_INC = -DUSE_HYPRE_AMG -I$(HYPRE_DIR)/include
HYPRE_LIB = $(HYPRE_DIR)/lib/libHYPRE.a
OTINUM_INC = -I/work/gyj329/MFEM/implementations/bridge_4layer_kokkos/MFEM_sam_examples/MFEM_examples/cpp_oti_lib/include

SRC = full_bridge_activation_tier1_oti.cpp

# Enzyme forward-mode kernels (clang + ClangEnzyme plugin); the driver itself is built by MFEM's g++.
ENZYME_ROOT   = /work/gyj329/MFEM/implementations/bridge_case_enzyme/third_party/enzyme_build
ENZYME_PLUGIN = $(firstword $(wildcard $(ENZYME_ROOT)/Enzyme/ClangEnzyme-*.so))
ENZ_CLANGXX   = /usr/lib64/llvm21/bin/clang++
ENZ_OBJ       = fb_enzyme_kernels.o
fb_enzyme_kernels.o: fb_enzyme_kernels.cpp fb_enzyme_kernels.h src_kernel.hpp
	$(ENZ_CLANGXX) -O2 -std=c++17 -fPIC -fno-exceptions -fplugin=$(ENZYME_PLUGIN) -DMFEM_USE_ENZYME -I$(MFEM_DIR) -c $< -o $@
SHAPES ?= 1x1 1x2 1x3 1x4 2x1 2x2 2x3 2x4 3x1 3x2 3x3 3x4 4x1 4x2 4x3 4x4 5x1 5x2 5x3 5x4 10x1 10x2
OBJ = obj
SHAPE_OBJS = $(patsubst %,$(OBJ)/eval_%.o,$(SHAPES))
AO_OBJS = $(OBJ)/eval_registry.o $(SHAPE_OBJS)
DEPS = fb_enzyme_kernels.h src_kernel.hpp tie_constraint.hpp pardiso_direct.hpp oti_type_generic30.hpp anyorder_eval.hpp query.hpp oti_query_type.hpp $(CONFIG_MK)

all: enzyme
enzyme: fbke_enzyme
order1: fbko_opt_order1
order2: fbko_opt_order2
# ONE Enzyme binary does -dorder 0/1/2 (no OTI arithmetic in the sensitivity blocks; OTI_ORDER_VAL=1 only sizes the leftover OTI statics)
fbke_enzyme: $(SRC) $(DEPS) $(AO_OBJS) $(ENZ_OBJ) fb_enzyme_kernels.h
	$(MFEM_CXX) $(MFEM_FLAGS) -std=c++17 -fopenmp -DUSE_ENZYME_AD -DOTI_ORDER_VAL=1 $(OTINUM_INC) $(PARDISO_INC) $(HYPRE_INC) -I. $< $(AO_OBJS) $(ENZ_OBJ) -o $@ $(MFEM_LIBS) $(HYPRE_LIB) $(PARDISO_LIB) -fopenmp
fbko_opt_order1: $(SRC) $(DEPS) $(AO_OBJS)
	$(MFEM_CXX) $(MFEM_FLAGS) -std=c++17 -fopenmp -DOTI_ORDER_VAL=1 $(OTINUM_INC) $(PARDISO_INC) $(HYPRE_INC) -I. $< $(AO_OBJS) -o $@ $(MFEM_LIBS) $(HYPRE_LIB) $(PARDISO_LIB) -fopenmp
fbko_opt_order2: $(SRC) $(DEPS) $(AO_OBJS)
	$(MFEM_CXX) $(MFEM_FLAGS) -std=c++17 -fopenmp -DOTI_ORDER_VAL=2 $(OTINUM_INC) $(PARDISO_INC) $(HYPRE_INC) -I. $< $(AO_OBJS) -o $@ $(MFEM_LIBS) $(HYPRE_LIB) $(PARDISO_LIB) -fopenmp
clean:
	rm -rf fbke_enzyme fbko_opt_order1 fbko_opt_order2 $(OBJ) shapes.inc fb_enzyme_kernels.o

shapes.inc: Makefile
	@rm -f $@; for s in $(SHAPES); do m=$${s%x*}; n=$${s#*x}; echo "SHAPE($$m,$$n)" >> $@; done
$(OBJ):
	mkdir -p $(OBJ)
$(OBJ)/eval_%.o: eval_shape.cpp anyorder_eval.hpp oti_query_type.hpp | $(OBJ)
	g++ -O3 -std=c++17 -fopenmp $(OTINUM_INC) -I. -DJET_M=$(word 1,$(subst x, ,$*)) -DJET_N=$(word 2,$(subst x, ,$*)) -c $< -o $@
$(OBJ)/eval_registry.o: eval_registry.cpp shapes.inc anyorder_eval.hpp | $(OBJ)
	g++ -O3 -std=c++17 -fopenmp $(OTINUM_INC) -I. -c $< -o $@
