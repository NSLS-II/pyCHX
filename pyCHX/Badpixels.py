"""Dev@Octo12,2017"""

import numpy as np

damaged_4Mpixel = np.array(
    [
        [1157, 2167 - 1231],
        [1158, 2167 - 1231],
        [1159, 2167 - 1231],
        [1160, 2167 - 1231],
        [1157, 2167 - 1230],
        [1158, 2167 - 1230],
        [1159, 2167 - 1230],
        [1160, 2167 - 1230],
        [1161, 2167 - 1230],
        [1157, 2167 - 1229],
        [1158, 2167 - 1229],
        [1159, 2167 - 1229],
        [1160, 2167 - 1229],
        [1159, 2167 - 1228],
        [1160, 2167 - 1228],
        [1159, 2167 - 1227],
        [1160, 2167 - 1227],
        [1159, 2167 - 1226],
    ]
)


# March 1, 2018
# uid = '92394a'
bad_pixel_4M = {
    "92394a": np.array(
        [
            828861,
            882769,
            915813,
            928030,
            959317,
            959318,
            992598,
            992599,
            998768,
            1009202,
            1036105,
            1143261,
            1149650,
            1259208,
            1321301,
            1426856,
            1426857,
            1586163,
            1774616,
            1936607,
            1936609,
            1936610,
            1938677,
            1938678,
            1938681,
            1940747,
            1946959,
            1955276,
            2105743,
            2105744,
            2107813,
            2107815,
            2109883,
            2118276,
            2118277,
            2149798,
            2194925,
            2283956,
            2284016,
            2284225,
            2284388,
            2290249,
            2292593,
            2298770,
            2304729,
            2317145,
            2344268,
            2346156,
            2356554,
            2360827,
            2364960,
            2408361,
            2453913,
            2470447,
            2476691,
            3462303,
            4155535,
        ]
    ),  # 57 points, coralpor
    "6cc34a": np.array([1058942, 2105743, 2105744, 2107813, 2107815, 2109883, 4155535]),  #  coralpor
}


## Create during 2018 Cycle 1
BadPix_4M = np.array(
    [
        828861,
        882769,
        915813,
        928030,
        959317,
        959318,
        992598,
        992599,
        998768,
        1009202,
        1036105,
        1143261,
        1149650,
        1259208,
        1321301,
        1426856,
        1426857,
        1586163,
        1774616,
        1936607,
        1936609,
        1936610,
        1938677,
        1938678,
        1938681,
        1940747,
        1946959,
        1955276,
        2105743,
        2105744,
        2107813,
        2107815,
        2109883,
        2118276,
        2118277,
        2149798,
        2194925,
        2283956,
        2284016,
        2284225,
        2284388,
        2290249,
        2292593,
        2298770,
        2304729,
        2317145,
        2344268,
        2346156,
        2356554,
        2360827,
        2364960,
        2408361,
        2453913,
        2470447,
        2476691,
        3462303,
        4155535,
        1058942,
        2105743,
        2105744,
        2107813,
        2107815,
        2109883,
        4155535,
        2107814,
        3462303,
    ]
)
