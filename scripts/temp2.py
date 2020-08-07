from shutil import copytree


i = "/SP-SLAM/g2oAddition"
o = "/slamdoom/tmp/orbslam2/g2oAddition"

copytree(i, o)