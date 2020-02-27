from unittest import TestCase
import importlib_resources
import tests.saved_test_data
from aspire.apple.apple import Apple


class ApplePickerTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testPickCenters(self):
        # 440 particles with the following centers
        centers = {
            (1261,  149), (2635,  173), (1715,  167), ( 249,  183), ( 715,  189), (2535,  201), (1025,  209),
            (1303,  215), (1453,  225), ( 555,  225), (1215,  241), (2165,  247), (3055,  237), ( 455,  257),
            (2303,  261), ( 291,  315), (1647,  273), (3129,  283), (1815,  301), ( 165,  329), (2453,  307),
            ( 939,  331), (1319,  361), ( 875,  373), (3131,  353), (1955,  367), ( 389,  367), (1717,  367),
            (1741,  375), (3211,  405), (3383,  401), (2879,  433), (1249,  451), ( 921,  493), (2649,  493),
            (3091,  497), ( 783,  509), (2411,  507), (1507,  529), (3229,  539), ( 545,  619), (1997,  551),
            (3317,  547), ( 747,  577), (1423,  613), (1645,  611), (2481,  615), ( 149,  621), (2969,  643),
            (1795,  671), (1197,  653), ( 889,  671), (2081,  667), (2697,  665), ( 465,  685), (3387,  719),
            (3113,  683), (3219,  709), (3109,  695), (3473,  705), ( 239,  733), ( 987,  709), (1687,  735),
            (2223,  713), (1777,  719), (2051,  781), (2197,  733), ( 763,  727), ( 769,  725), (2337,  757),
            ( 153,  735), ( 403,  771), (1381,  811), (1207,  789), (3463,  821), (1691,  819), (2175,  819),
            (3127,  849), (1169,  845), (1509,  847), (1905,  861), (1819,  865), (2937,  871), (3573,  861),
            ( 305,  883), (2273,  935), (1703,  885), (2397,  885), ( 959,  881), (2099,  963), (1989,  919),
            (2473,  909), ( 159,  981), (1585,  929), ( 371,  953), (1139,  931), (1233,  941), (3199,  959),
            (1163,  999), (1441,  985), (3581,  989), (3583,  993), (1905, 1029), (3681, 1029), ( 547, 1041),
            (2621, 1065), (3171, 1071), (3035, 1089), (2117, 1079), (2897, 1081), (3433, 1067), (1019, 1075),
            (1363, 1111), ( 355, 1117), (1463, 1137), ( 475, 1131), ( 485, 1129), (3241, 1139), (2725, 1181),
            (2027, 1167), (3807, 1191), ( 873, 1199), (2547, 1275), (2793, 1237), (3695, 1231), (3267, 1237),
            (1693, 1253), ( 347, 1255), (3121, 1267), (2957, 1267), ( 535, 1287), ( 157, 1299), (1905, 1303),
            (1789, 1289), (1801, 1287), (2189, 1339), (2267, 1331), (1277, 1339), (3677, 1337), ( 255, 1369),
            (1483, 1365), (3159, 1395), ( 945, 1397), (1739, 1393), (2647, 1427), (3837, 1421), (2461, 1439),
            (3487, 1453), ( 485, 1459), (3085, 1461), (3209, 1455), ( 293, 1459), (3571, 1457), (3863, 1457),
            (3219, 1459), (3225, 1459), (3729, 1467), (3899, 1489), (1443, 1491), (3469, 1497), (1473, 1555),
            (1951, 1551), ( 991, 1535), (2281, 1541), (1649, 1545), (3807, 1547), (1685, 1557), ( 785, 1587),
            (1689, 1559), ( 387, 1579), (1239, 1571), ( 417, 1569), (3031, 1607), (1115, 1607), (2699, 1629),
            (3695, 1639), (3893, 1647), (1009, 1657), (1547, 1649), (1115, 1687), (1987, 1693), (3285, 1681),
            (1333, 1689), (3581, 1727), ( 165, 1697), (1511, 1719), ( 325, 1717), (2703, 1735), ( 755, 1773),
            (2765, 1729), (3363, 1757), ( 983, 1751), (3075, 1807), (2359, 1857), (1955, 1823), ( 859, 1839),
            (3409, 1843), (3483, 1849), ( 939, 1849), ( 983, 1863), (2487, 1875), ( 357, 1885), (2825, 1903),
            (2627, 1907), (1575, 1915), (1549, 1957), ( 255, 1923), ( 523, 1959), ( 857, 1947), (1911, 1939),
            (1387, 1967), (3855, 1965), (1243, 1977), (2829, 2011), (2517, 1975), (1665, 2003), (3475, 1987),
            ( 883, 2015), ( 809, 2031), ( 547, 2055), (2655, 2043), (3611, 2047), (3289, 2097), (3185, 2113),
            (3581, 2113), (1233, 2137), (3581, 2119), (3401, 2145), (3489, 2187), ( 223, 2181), (2439, 2177),
            (1447, 2201), (2927, 2253), (2215, 2207), (2835, 2219), ( 289, 2219), (3657, 2237), (2245, 2259),
            (1257, 2275), (3735, 2253), (2977, 2255), ( 817, 2327), (1811, 2281), (3367, 2281), (3539, 2295),
            ( 965, 2307), (2865, 2291), (3091, 2327), (2245, 2327), (1361, 2351), (3779, 2341), (3909, 2347),
            (2707, 2367), (1017, 2365), (1965, 2395), (3363, 2387), (1245, 2381), (1663, 2361), ( 669, 2387),
            (1529, 2387), (1653, 2407), (3921, 2415), ( 547, 2421), ( 437, 2427), (3577, 2457), (2243, 2467),
            ( 749, 2475), ( 317, 2475), (3419, 2479), (1765, 2507), (2143, 2487), (1111, 2497), (1225, 2521),
            (3475, 2509), (1145, 2515), (3379, 2535), ( 691, 2533), (2935, 2543), (1367, 2553), (2415, 2563),
            (2937, 2593), (2693, 2587), (3271, 2593), ( 393, 2591), (1887, 2593), ( 529, 2609), (3339, 2601),
            ( 255, 2601), (2601, 2595), ( 813, 2617), (1997, 2607), (2509, 2615), ( 699, 2625), (1777, 2631),
            (3935, 2623), ( 895, 2643), (3907, 2641), (1865, 2645), ( 693, 2661), (1733, 2663), (1963, 2675),
            (3251, 2703), (2381, 2683), (1733, 2679), (3439, 2685), ( 215, 2717), (2769, 2719), (1495, 2737),
            (1083, 2735), (2153, 2743), (3577, 2741), (2683, 2751), (1087, 2765), (1741, 2757), (2789, 2779),
            (1815, 2793), (1249, 2823), (1533, 2855), (1923, 2819), (2021, 2833), (2747, 2823), (1367, 2847),
            (3523, 2825), (3321, 2831), (3529, 2851), (3651, 2847), (3075, 2871), ( 851, 2877), (1907, 2855),
            (2359, 2871), ( 597, 2943), ( 533, 2889), ( 485, 2887), (3507, 2899), ( 959, 2901), (3523, 2905),
            ( 387, 2933), (3131, 2945), (3933, 2957), (1585, 2975), ( 841, 2983), (2179, 3037), ( 821, 3011),
            (1251, 3039), (2559, 3045), (1943, 3053), (2297, 3053), (1371, 3055), (2469, 3051), ( 957, 3061),
            (2777, 3061), (3877, 3063), ( 853, 3077), ( 719, 3071), ( 279, 3079), (3783, 3089), ( 439, 3095),
            (1483, 3091), ( 531, 3149), ( 691, 3101), (2387, 3125), (1219, 3149), (2977, 3151), (2299, 3167),
            (1737, 3193), (2037, 3283), ( 715, 3201), ( 993, 3197), (3901, 3227), (2163, 3247), ( 861, 3253),
            ( 499, 3301), (1345, 3311), ( 225, 3245), (1941, 3299), ( 747, 3299), (3221, 3295), (1495, 3309),
            (1235, 3297), (3335, 3315), (2727, 3333), (1077, 3315), (2287, 3335), (3107, 3341), (1191, 3375),
            (1705, 3403), ( 343, 3379), (3351, 3393), (1869, 3385), (3763, 3403), ( 827, 3401), (2769, 3405),
            (1805, 3423), (3079, 3441), ( 925, 3509), (1187, 3457), (3367, 3463), ( 379, 3471), (3007, 3499),
            (1917, 3529), (2501, 3491), ( 725, 3491), (2327, 3477), (3331, 3485), (2391, 3509), (1087, 3501),
            (2701, 3553), (3593, 3533), (2011, 3561), ( 583, 3559), (2581, 3575), (2131, 3587), (1565, 3571),
            (3435, 3577), (1579, 3577), (3763, 3583), (3943, 3579), (2403, 3619), ( 505, 3611), ( 697, 3609),
            (1023, 3621), ( 355, 3623), (3593, 3653), ( 327, 3673), (3273, 3685), (2797, 3701), (1127, 3697),
            (3819, 3711), (2657, 3719), (2829, 3753), (3059, 3739), (3159, 3763), ( 525, 3769), (2993, 3771),
            (1169, 3799), (1507, 3785), (1733, 3781), (2655, 3847), (2053, 3861), (2855, 3883)
        }

        apple_picker = Apple()

        with importlib_resources.path(tests.saved_test_data, 'sample.mrc') as mrc_path:
            centers_found = apple_picker.process_micrograph(mrc_path)
            for center_found in centers_found:
                _x, _y = tuple(center_found)
                if (_x, _y) not in centers:
                    self.fail('({}, {}) not an expected center.'.format(_x, _y))
                else:
                    centers.remove((_x, _y))

            if centers:
                self.fail('Not all expected centers were found!')
