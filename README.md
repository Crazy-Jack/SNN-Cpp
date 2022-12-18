# SNN-Cpp
Spike Neural Network training Implemented directly using PyTorch Cpp Frontend (libtorch)

# Install Pytorch C++ library -- libtorch
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

After CMake is updated, build project with the library pointed as follows:
```
cmake -DCMAKE_PREFIX_PATH=/home/tianqinl/SNN-Cpp/cache_replacement/libtorch ..
```
See [`build_binary.sh`](cache_replacement/build_binary.sh) for how to compile with libtorch properly.

# Build
```
export CC=`which gcc`
export CXX=`which g++`

rm -rf build
mkdir build
cd build 
cmake -DCMAKE_PREFIX_PATH=/home/tianqinl/SNN-Cpp/cache_replacement/libtorch ..
cmake --build . --config Release
cd ..
```


# RUN Cache Champ 2

See [doc](cache_replacement/ChampCache/README.txt) for details.



