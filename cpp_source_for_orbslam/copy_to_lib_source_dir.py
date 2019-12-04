from glob import glob
import os
from shutil import copyfile

src_dir = "/slamdoom/tmp/orbslam2/src/"
inc_dir = "/slamdoom/tmp/orbslam2/include/"


fs_cc = glob(os.path.dirname(__file__) + "/*.cc")
fs_h = glob(os.path.dirname(__file__) + "/*.h")

# for f in fs_cc:
#     fname = f.split("/")[-1]
#     o = src_dir + "/" + fname
#     copyfile(f, o)
#
#
# for f in fs_h:
#     fname = f.split("/")[-1]
#     o = inc_dir + "/" + fname
#     copyfile(f, o)

## copy_back
for f in fs_cc:
    fname = f.split("/")[-1]
    i = src_dir + "/" + fname
    o = f + ".backup"
    copyfile(i, o)

for f in fs_h:
    fname = f.split("/")[-1]
    i = inc_dir + "/" + fname
    o = f + ".backup"
    copyfile(i, o)
