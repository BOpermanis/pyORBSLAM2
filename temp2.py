import os
from pprint import pprint

walker = os.walk("/home/slam_data/SP-SLAM/Thirdparty/")
for d, ds, fs in walker:
    for f in fs:
        try:
            with open(d + "/" + f, 'r') as conn:
                lns = conn.read()
        except:
            print(f)
            continue
        # pprint(lns)
        #if '#include "../../config.h"' in lns:
        if '#include "../../Config.h"' in lns:
            print(d + "/" + f)
            lns = lns.replace('#include "../../config.h"', '//#include "Config.h"')
            with open(d + "/" + f, 'w') as conn:
                conn.write(lns)
        # print(type(lns))
        # exit()
# print(next(iter(walker)))
