CPP_DIR=cpp_code
CYT=crosscat/cython_code
DOC=docs
TEST=$(CPP_DIR)/tests
XNET=crosscat/binary_creation


all: cython doc

clean:
	cd $(CPP_DIR) && make clean
	cd $(CYT) && make clean
	cd $(DOC) && make clean
	cd $(TEST) && make clean
	cd $(XNET) && make clean

cpp:
	cd $(CPP_DIR) && make

cython:
	cd $(CYT) && make

doc:
	cd $(DOC) && make

runtests:
	cd $(CPP_DIR) && make runtests

tests:
	cd $(CPP_DIR) && make tests

xnet:
	cd $(XNET) && make
