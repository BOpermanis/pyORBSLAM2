import os
from pprint import pprint

walker = os.walk("/SP-SLAM/Thirdparty/")
for d, ds, fs in walker:
    for f in fs:
        with open(d + "/" + f, 'r') as conn:
            lns = conn.read()
        # pprint(lns)
        if '#include "../../config.h"' in lns:
            print(d + "/" + f)
            lns = lns.replace('#include "../../config.h"', '//#include "Config.h"')
            with open(d + "/" + f, 'w') as conn:
                conn.write(lns)
        # print(type(lns))
        # exit()
# print(next(iter(walker)))