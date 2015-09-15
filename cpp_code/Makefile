OPTIMIZED = True
ifdef OPTIMIZED
CXXOPTS := -pg -O2 $(CXXOPTS)
else
CXXOPTS := -pg -O0 -g $(CXXOPTS)
endif

CC = gcc
CXX = g++
CXXOPTS := $(CXXOPTS) -lm

ifdef BOOST_ROOT
CXXOPTS := $(CXXOPTS) -L$(BOOST_ROOT)/lib -I$(BOOST_ROOT)/include
endif

INC=include/CrossCat
SRC=src
OBJ=obj
TEST=tests
BIN = main
MAIN = main.cpp
NAMES = \
	Cluster \
	ComponentModel \
	ContinuousComponentModel \
	CyclicComponentModel \
	DateTime \
	MultinomialComponentModel \
	RandomNumberGenerator \
	State \
	View \
	numerics \
	utils \
	# end of NAMES
TEST_NAMES = \
	test_cluster \
	test_component_model \
	test_continuous_component_model \
	test_multinomial_component_model \
	test_numerics \
	# end of TEST_NAMES
BROKEN_TEST_NAMES = \
	test_state \
	test_view \
	test_view_speed \
	# end of BROKEN_TEST_NAMES
HEADERS = $(foreach name, $(NAMES), $(INC)/$(name).h)
SOURCES = $(foreach name, $(NAMES), $(SRC)/$(name).cpp)
OBJECTS = $(foreach name, $(NAMES), $(OBJ)/$(name).o)
TESTS = $(foreach test, $(TEST_NAMES), $(TEST)/$(test))

all: obj # bin

tests: obj $(TESTS)

# run each test
runtests: obj $(TESTS)
	@echo tests are: $(TESTS) $(foreach test, $(TESTS), && ./$(test))

clean:
	rm -f $(BIN) $(OBJECTS) core *.stackdump *.gch $(TESTS)

bin: $(OBJECTS) $(BIN) $(MAIN) $(HEADERS)

obj: $(OBJECTS)

$(OBJ)/%.o: $(SRC)/%.cpp $(INC)/%.h $(HEADERS)
	$(CXX) -c $< -o $@ $(CXXOPTS) -I$(INC)

$(BIN): $(OBJECTS) $(MAIN) $(HEADERS)
	$(CXX) -o $(BIN) $(MAIN) $(OBJECTS) $(HEADERS) $(CXXOPTS) -lboost_program_options -I$(INC)

$(TEST)/%: $(TEST)/%.cpp $(HEADERS) $(OBJECTS)
	$(CXX) $< -o $@ $(CXXOPTS) -I$(INC) $(OBJECTS)
