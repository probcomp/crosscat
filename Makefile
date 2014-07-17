CPP_DIR=cpp_code
CYT=crosscat/cython_code
DOCS=docs
TEST=$(CPP_DIR)/tests
XNET=crosscat/binary_creation


all: cython docs

clean:
	cd $(CPP_DIR) && make clean
	cd $(CYT) && make clean
	cd $(XNET) && make clean
	#
	cd $(DOCS)/sphinx && rm -rf _build
	cd $(DOCS)/latex && make clean
	cd $(DOCS)/doxygen && rm -rf html latex

cpp:
	cd $(CPP_DIR) && make

cython:
	cd $(CYT) && make

docs:	cython
	cd $(DOCS)/sphinx && make -f Makefile.sphinx html latexpdf
	cd $(DOCS)/latex && make
	cd $(DOCS)/doxygen && doxygen Doxyfile && cd latex && make

runtests:
	cd $(CPP_DIR) && make runtests

tests:
	cd $(CPP_DIR) && make tests

xnet:
	cd $(XNET) && make
