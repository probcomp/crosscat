CXXOPTS := $(CXXOPTS) -Wall -Werror -std=c++98
OPTIMIZED = True
ifdef OPTIMIZED
CXXOPTS := -O2 $(CXXOPTS)
else
CXXOPTS := -O0 -g $(CXXOPTS)
endif

CC = gcc
CXX = g++
CXXOPTS := $(CXXOPTS)

INC=include/CrossCat
SRC=src
OBJ=obj
TEST=tests
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
	weakprng \
	# end of NAMES
TEST_NAMES = \
	test_cluster \
	test_component_model \
	test_continuous_component_model \
	test_matrix \
	test_multinomial_component_model \
	test_numerics \
	test_random_number_generator \
	test_utils \
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

all: obj

tests: obj $(TESTS)

# run each test
runtests: obj $(TESTS)
	@echo tests are: $(TESTS) $(foreach test, $(TESTS), && ./$(test))

clean:
	rm -f $(OBJECTS) core *.stackdump *.gch $(TESTS)

obj: $(OBJECTS)

$(OBJ)/%.o: $(SRC)/%.cpp $(INC)/%.h $(HEADERS)
	$(CXX) -c $< -o $@ $(CXXOPTS) -I$(INC)

$(TEST)/%: $(TEST)/%.cpp $(HEADERS) $(OBJECTS)
	$(CXX) $< -o $@ $(CXXOPTS) -I$(INC) $(OBJECTS)
