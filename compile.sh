cd hash_table
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

cd ../../neighbor
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

cd ../../bitmap
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

cd ../../work_stealing
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ../..