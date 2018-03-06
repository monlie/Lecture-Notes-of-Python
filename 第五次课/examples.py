# -*- coding: utf-8 -*-
"""
Created on 2017/11/10 18:55:34

@author: 李蒙
"""

"""
1. is_prime(n)            判断是否为素数 
2. which_day(y, m, d)     给出年月日判断第几天 
3. all_prime(n)           寻找小于n的所有素数 
4. newton_method(f)       牛顿迭代法 
5. calc_pi(n)             迭代计算圆周率 
6. bubble_sort(l)         冒泡排序
"""

import unittest
from math import sqrt


def is_prime(n):
    i = 2
    while i <= sqrt(n):
        if n % i == 0:
            return False
        i += 1
    return True


def which_day(y, m, d):
    month = {1: 31,
             2: 28,
             3: 31,
             4: 30,
             5: 31,
             6: 30,
             7: 31,
             8: 31,
             9: 30,
             10: 31,
             11: 30,
             12: 31}
    if y % 4 == 0:
        month[2] = 29
    s = 0
    for i in range(1, m):
        s += month[i]
    return s+d


# 筛法求所有不大于n的素数
def all_prime(n):
    pass


def bubble_sort(l):
    length = len(l)
    while length:
        for i in range(length-1):
            if l[i] > l[i+1]:
                l[i], l[i+1] = l[i+1], l[i]
        length -= 1
    return l


# 软件工程的思想：TDD（测试驱动开发）
class TestAll(unittest.TestCase):

    def test_prime(self):
        p = is_prime
        self.assertEqual(p(33), False)
        self.assertEqual(p(101), True)
        self.assertEqual(p(121), False)
        self.assertEqual(p(320150927951), True)
        self.assertEqual(p(320150928051), False)

    def test_day(self):
        d = which_day
        self.assertEqual(d(2004, 12, 31), 366)
        self.assertEqual(d(2003, 12, 31), 365)
        self.assertEqual(d(2016, 3, 1), 61)
        self.assertEqual(d(2017, 3, 5), 64)

    def test_all_prime(self):
        a = all_prime
        self.assertEqual(a(10), [2, 3, 5, 7])
        self.assertEqual(a(20), [2, 3, 5, 7, 11, 13, 17, 19])
        self.assertEqual(a(35), [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])

    def test_sort(self):
        s = bubble_sort
        self.assertEqual(s([3, 1, 2]), [1, 2, 3])
        self.assertEqual(s([1, 3, 3, 0, -2, 7]), [-2, 0, 1, 3, 3, 7])


if __name__ == '__main__':
    unittest.main()
