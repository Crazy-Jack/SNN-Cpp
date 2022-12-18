export CC=`which gcc`
export CXX=`which g++`

rm -rf build
mkdir build
cd build 
cmake -DCMAKE_PREFIX_PATH=/home/tianqinl/SNN-Cpp/src/libtorch ..
cmake --build . --config Release
cd ..

./build/snn-cpp -warmup_instructions 1000000 -simulation_instructions 10000000 -traces ChampCache/trace/bzip2_10M.trace.gz