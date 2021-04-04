$(info "mkl start")
URL=https://github.com/intel/mkl-dnn/archive/v1.6.1.tar.gz
ifndef WGET
WGET = wget
endif

ifndef MKL_PATH
MKL_PATH = $(shell pwd)/mkl-dnnl/build
endif

MKL_DNNL = ${MKL_PATH}/include/dnnl.hpp

${MKL_DNNL}:
	$(eval FILE=v1.6.1.tar.gz)
	$(eval DIR=oneDNN-1.6.1)
	rm -rf $(FILE) $(DIR)
	$(info ${FILE})
	$(WGET) $(URL) && tar --no-same-owner -zxf $(FILE)
	cd $(DIR) && mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=$(MKL_PATH) .. && $(MAKE) && $(MAKE) install
	rm -rf $(FILE) $(DIR)