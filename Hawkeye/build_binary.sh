export CC=`which gcc`
export CXX=`which g++`

rm -rf build
mkdir build
cd build 
cmake ..
cmake --build . --config Release
cd ..


./build/hawkeye -warmup_instructions 1000000 -simulation_instructions 10000000 -traces trace/bzip2_10M.trace.gz