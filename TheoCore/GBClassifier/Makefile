# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall -Iinclude

# SQLite3 library location
LDFLAGS = -lsqlite3

# Source files
SRC = src/main.cpp src/gbc.cpp src/data_loader.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Output binary
TARGET = gbc_app

# Build rules
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -o $(TARGET) $(OBJ) $(LDFLAGS)

# Compile each source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJ) $(TARGET)