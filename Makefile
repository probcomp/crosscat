CPP_DIR=cpp_code
CYT=crosscat/cython_code
DOC=docs
TEST=$(CPP_DIR)/tests
XNET=crosscat/binary_creation


all: cython doc

clean:
	cd $(CPP_DIR) && make clean
	cd $(CYT) && make clean
	cd $(XNET) && make clean
	#
	cd $(DOC)/sphinx && rm -rf _build
	cd $(DOC)/latex && make clean
	cd $(DOC)/doxygen && rm -rf html latex

cpp:
	cd $(CPP_DIR) && make

cython:
	cd $(CYT) && make

doc:
	cd $(DOC)/sphinx && make -f Makefile.sphinx
	cd $(DOC)/latex && make
	cd $(DOC)/doxygen && doxygen Doxyfile

runtests:
	cd $(CPP_DIR) && make runtests

tests:
	cd $(CPP_DIR) && make tests

xnet:
	cd $(XNET) && make
