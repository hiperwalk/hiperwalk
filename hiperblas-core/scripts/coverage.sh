#!/bin/bash

# Set the name of your project
PROJECT_NAME="coverage"

# Set the output directory for the coverage report
COVERAGE_DIR="coverage"

# Clean up previous coverage data
lcov --directory . --zerocounters

# Build your project with CMake
cmake -DCMAKE_BUILD_TYPE=Debug -DCODE_COVERAGE=true ..
make 
sudo make install
make test

# Capture coverage data
lcov --directory . --capture --output-file $PROJECT_NAME.info

# Filter out system and external dependencies (modify as needed)
lcov --remove $PROJECT_NAME.info "/usr/*" --output-file $PROJECT_NAME.info
lcov --remove $PROJECT_NAME.info "*/external/*" --output-file $PROJECT_NAME.info

# Generate HTML report
genhtml $PROJECT_NAME.info --output-directory $COVERAGE_DIR

# Display the report in a web browser
xdg-open $COVERAGE_DIR/index.html
