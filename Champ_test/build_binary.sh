export CC=`which gcc`
export CXX=`which g++`

rm -rf build
mkdir build
cd build 
cmake ..
cmake --build . --config Release
cd ..
