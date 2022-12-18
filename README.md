# SNN-Cpp
Spike Neural Network training Implemented directly using PyTorch Cpp Frontend (libtorch)

# Install C++ Distributions of Pytorch
[https://pytorch.org/cppdocs/installing.html#minimal-example](https://pytorch.org/cppdocs/installing.html#minimal-example)


When using Cmake, if CMake is not up to date, download the source code from [https://cmake.org/download/](https://cmake.org/download/) and install it locally.
```
tar -xf cmake*.tar.gz

cd cmake*

./configure --prefix=$HOME

make

make install

You should now have the most up-to-date installation of cmake. Check the version by typing:

cmake --version
```

# Build
```
export CC=`which gcc`
export CXX=`which g++`

rm -rf build
mkdir build
cd build 
cmake -DCMAKE_PREFIX_PATH=/home/tianqinl/SNN-Cpp/example-app/libtorch ..
cmake --build . --config Release
cd ..
```



