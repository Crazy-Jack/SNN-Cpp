export CC=`which gcc`
export CXX=`which g++`

rm -rf build
mkdir build
cd build 
cmake -DCMAKE_PREFIX_PATH=/home/tianqinl/SNN-Cpp/NN/libtorch ..
cmake --build . --config Release
cd ..

./build/simple