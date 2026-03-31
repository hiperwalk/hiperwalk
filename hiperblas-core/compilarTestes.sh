export LD_LIBRARY_PATH=~/hiperblas/lib/
srcName=vectorTest
g++  test/$srcName.cpp -I ~/hiperblas/include/ -L ~/hiperblas/lib/ -lhiperblas-core -lhiperblas-cpu-bridge -lgtest -lm -lgtest_main -o $srcName
./$srcName

srcName=sparseMatrixTest
g++  test/$srcName.cpp -I ~/hiperblas/include/ -L ~/hiperblas/lib/ -lhiperblas-core -lhiperblas-cpu-bridge -lgtest -lm -lgtest_main -o $srcName
./$srcName
