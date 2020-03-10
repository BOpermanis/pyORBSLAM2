cd /slamdoom/tmp/orbslam2/
./build.sh || exit 1

cd /home/slam_data/pyORBSLAM2/src
./build.sh || exit 1


export PYTHONPATH=/home/slam_data/pyORBSLAM2/src/build:$PYTHONPATH
cd /home/slam_data/pyORBSLAM2/scripts
#python3 test_slam_with_webcam.py
#python3 temp.py


