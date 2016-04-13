CPP_DIR=cpp_code
CYT=crosscat/cython_code
DOCS=docs
TEST=$(CPP_DIR)/tests


all: cython docs

clean:
	cd $(CPP_DIR) && $(MAKE) clean
	cd $(CYT) && $(MAKE) clean
	cd $(DOCS)/sphinx && rm -rf _build
	cd $(DOCS)/latex && $(MAKE) clean
	cd $(DOCS)/doxygen && rm -rf html latex

cpp:
	cd $(CPP_DIR) && $(MAKE)

cython:
	cd $(CYT) && $(MAKE)

docs:	cython
	cd $(DOCS)/sphinx && $(MAKE) -f Makefile.sphinx html latexpdf
	cd $(DOCS)/latex && $(MAKE)
	cd $(DOCS)/doxygen && doxygen Doxyfile && cd latex && $(MAKE)

runtests:
	cd $(CPP_DIR) && $(MAKE) runtests

tests:
	cd $(CPP_DIR) && $(MAKE) tests
