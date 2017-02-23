from random import choice, random


def lcompress(row):
    lo, hi = 0, 1
    changed = False
    while True:
        while hi < 4 and not row[hi]: hi += 1
        if hi == 4: return changed
        if not row[lo]:
            row[lo] = row[hi]
            row[hi] = 0
            hi += 1
            changed = True
        elif row[lo] == row[hi]:
            row[hi] = 0
            row[lo] += 1
            hi += 1
            lo += 1
            changed = True
        elif hi > lo + 1:
            row[lo + 1] = row[hi]
            row[hi] = 0
            hi += 1
            lo += 1
            changed = True
        else:
            hi += 1
            lo += 1


def rcompress(row):
    lo, hi = 3, 2
    changed = False
    while True:
        while hi > -1 and not row[hi]: hi -= 1
        if hi == -1: return changed
        if not row[lo]:
            row[lo] = row[hi]
            row[hi] = 0
            hi -= 1
            changed = True
        elif row[lo] == row[hi]:
            row[hi] = 0
            row[lo] += 1
            hi -= 1
            lo -= 1
            changed = True
        elif hi < lo - 1:
            row[lo - 1] = row[hi]
            row[hi] = 0
            hi -= 1
            lo -= 1
            changed = True
        else:
            hi -= 1
            lo -= 1


def lcheck(row):
    lo, hi = 0, 1
    while True:
        while hi < 4 and not row[hi]: hi += 1
        if hi == 4:
            return False
        elif not row[lo] or row[lo] == row[hi] or hi > lo + 1:
            return True
        else:
            hi += 1
            lo += 1


def rcheck(row):
    lo, hi = 3, 2
    while True:
        while hi > -1 and not row[hi]: hi -= 1
        if hi == -1:
            return False
        elif not row[lo] or row[lo] == row[hi] or hi < lo - 1:
            return True
        else:
            hi -= 1
            lo -= 1



def u(arr):
    arr = arr.T
    lcompress(arr[0])
    lcompress(arr[1])
    lcompress(arr[2])
    lcompress(arr[3])


def d(arr):
    arr = arr.T
    rcompress(arr[0])
    rcompress(arr[1])
    rcompress(arr[2])
    rcompress(arr[3])


def l(arr):
    lcompress(arr[0])
    lcompress(arr[1])
    lcompress(arr[2])
    lcompress(arr[3])

def r(arr):
    rcompress(arr[0])
    rcompress(arr[1])
    rcompress(arr[2])
    rcompress(arr[3])


def lable(arr):
    return(lcheck(arr[0])
        or lcheck(arr[1])
        or lcheck(arr[2])
        or lcheck(arr[3]))


def rable(arr):
    return(rcheck(arr[0])
        or rcheck(arr[1])
        or rcheck(arr[2])
        or rcheck(arr[3]))


def uable(arr):
    arr = arr.T
    return(lcheck(arr[0])
        or lcheck(arr[1])
        or lcheck(arr[2])
        or lcheck(arr[3]))


def dable(arr):
    arr = arr.T
    return(rcheck(arr[0])
        or rcheck(arr[1])
        or rcheck(arr[2])
        or rcheck(arr[3]))


def addtile(arr):
    l = []
    for i in range(4):
        for j in range(4):
            if arr[i, j]: continue
            l.append((i, j))
    arr[choice(l)] = 1 if random() < 0.9 else 2


