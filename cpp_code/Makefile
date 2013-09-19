OPTIMIZED = True
ifdef OPTIMIZED
CXXOPTS := -pg -O2 $(CXXOPTS)
else
CXXOPTS := -pg -O0 -g $(CXXOPTS)
endif

CC = gcc
CXX = g++
CXXOPTS :=  $(CXXOPTS) -lm -L$(BOOST_ROOT)/lib -I$(BOOST_ROOT)/include

INC=include/CrossCat
SRC=src
OBJ=obj
TEST=tests
# Assume BOOST_ROOT set as environment variable.
# /usr/local/boost
BIN = main
MAIN = main.cpp
NAMES = ContinuousComponentModel MultinomialComponentModel ComponentModel \
	Cluster View State \
	utils numerics RandomNumberGenerator DateTime
TEST_NAMES = test_component_model \
	test_continuous_component_model test_multinomial_component_model \
	test_cluster test_view test_view_speed test_state
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
	$(CXX) -o $(BIN) $(MAIN) $(OBJECTS) $(HEADERS) $(CXXOPTS) -I$(INC)

$(TEST)/%: $(TEST)/%.cpp $(HEADERS) $(OBJECTS)
	$(CXX) $< -o $@ $(CXXOPTS) -I$(INC) $(OBJECTS)
