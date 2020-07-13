#!/usr/bin/env bash

cmake -DBUILD_PYTHON3=ON \
      -DCMAKE_MODULE_PATH=/usr/local/include/eigen/cmake \
      -DZMQ_INCLUDE_DIR=/usr/local/include \
      -DZMQ_LIBRARY=/usr/local/lib/libzmq.so \
      -DORBSLAM2_LIBRARY=/SP-SLAM/lib/libORB_SLAM2.so \
      -DBG2O_LIBRARY=/SP-SLAM/Thirdparty/g2o/lib/libg2o.so \
      -DDBoW2_LIBRARY=/SP-SLAM/Thirdparty/DBoW2/lib/libDBoW2.so \
      -DCMAKE_BUILD_TYPE=Release .

export CPATH=/usr/local/include/eigen/:/SP-SLAM:/slamdoom/libs/cppzmq:/usr/local/include/

export LD_LIBRARY_PATH=/SP-SLAM/Thirdparty/DBoW2/lib

make -j2 --always-make
