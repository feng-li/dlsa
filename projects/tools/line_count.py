#! /usr/bin/env python3

import sys

count = 0
for line in sys.stdin:  # read input from stdin
    count += 1
print(count)  # print goes to sys.stdout
