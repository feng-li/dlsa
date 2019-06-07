#! /usr/bin/env python3

import os

print(os.path.dirname(os.path.abspath(__file__)))


arg1 = "test arg"

def x(arg2):
    if arg2 > 0:
        print(arg1)


x(5)
