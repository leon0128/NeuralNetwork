# program settings
## program name
PROGRAM = test
## source directories(not included last '/')
DIRECTORIES = $(SOURCE_DIRECTORY) \
	$(NEURAL_NETWORK_DIRECTORY) \
	$(MATRIX_DIRECTORY)

SOURCE_DIRECTORY         = ./src
NEURAL_NETWORK_DIRECTORY = ./src/neural_network
MATRIX_DIRECTORY         = ./src/matrix

# compiler settings
## c++ compiler
CXX = g++
## c++ compiler flags
CXXFLAGS = -g3 -Wall -std=c++2a -I./src

## linkage flags
LINKAGE_FLAGS = -lpthread -pthread

###########################################################
# files
## source files
SOURCE_FILES = $(foreach DIRECTORY, \
	$(DIRECTORIES),                 \
	$(wildcard $(DIRECTORY)/*.cpp))
## object files
OBJECT_FILES = $(patsubst %.cpp, \
	%.o,                         \
	$(SOURCE_FILES))
## dependency files
DEPENDENCY_FILES = $(patsubst %.cpp, \
	%.d,                             \
	$(SOURCE_FILES))

# recipes
## all
.PHONY: all
all: $(PROGRAM)

## $(PROGRAM) recipe
$(PROGRAM): $(OBJECT_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LINKAGE_FLAGS)

## dependency file recipe
.PHONY: $(DEPENDENCY_FILES)
$(DEPENDENCY_FILES):
	$(CXX) -E $(CXXFLAGS) -MM -MF $@ -MT $(basename $@).o $(basename $@).cpp

## clean
.PHONY: clean
clean:
	rm -rf $(PROGRAM)        \
		$(foreach DIRECTORY, \
			$(DIRECTORIES),  \
			$(wildcard $(DIRECTORY)/*[.o,.d]))

# include depndency files
ifeq ($(MAKECMDGOALS),)
include $(DEPENDENCY_FILES)
else ifeq ($(MAKECMDGOALS), all)
include $(DEPENDENCY_FILES)
else ifeq ($(MAKECMDGOALS), $(PROGRAM))
include $(DEPENDENCY_FILES)
else ifeq ($(suffix $(MAKECMDGOALS)), .o)
include $(basename $(MAKECMDGOALS)).d
else ifeq ($(MAKECMDGOALS), clean)
endif