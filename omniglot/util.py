
import os
import numpy as np
import numpy.random as rd
import PIL as pillow
import PIL.Image as image
import PIL.ImageOps as ops
import PIL.ImageFilter as filter

# from stackoverflow
def findcoeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

"radial warping with bilinear interpolation"
def warp(image, out):
    img = image.copy()
    imgw, imgh = image.shape
    w, h = out
    wl = np.linspace(-0.5, 0.5, w)
    hl = np.linspace(-0.5, 0.5, h)
    xa, ya = np.meshgrid(wl, hl)
    xa = xa.reshape([-1])
    ya = ya.reshape([-1])

    for i in range(rd.choice(3) + 1):
        x_shift, y_shift = rd.uniform(-0.4, 0.4, size=2)
        radius = rd.uniform(0, 0.5) ** 2
        strength = rd.uniform(-1, 1)

        indx = np.array([xa, ya]).T
        indx = indx.T 
        indx[0] += x_shift
        indx[1] += y_shift
        dist = (indx * indx).sum(axis=0)
        # inverted sigmoid
        scale = strength * np.exp(-dist / radius) + 1
        indx = indx * scale
        indx[0] -= x_shift - 0.5
        indx[1] -= y_shift - 0.5
        
        indx[0] *= imgw
        indx[1] *= imgh
        
        # bilinear interpolation
        # because it's easy and it works
        x, y = indx
        x0 = x.astype(int)
        y0 = y.astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        
        area00 = (x - x0)*(y - y0)
        area01 = (x - x0)*(y1 - y)
        area10 = (x1 - x)*(y - y0)
        area11 = (x1 - x)*(y1 - y)
        
        x  = np.clip(x, 0, imgw - 1)
        y  = np.clip(y, 0, imgh - 1)
        x0 = np.clip(x0, 0, imgw - 1)
        y0 = np.clip(y0, 0, imgh - 1)
        x1 = np.clip(x1, 0, imgw - 1)
        y1 = np.clip(y1, 0, imgh - 1)
        img = (area11 * img[y0, x0]
             + area10 * img[y1, x0]
             + area01 * img[y0, x1]
             + area00 * img[y1, x1])
        imgw, imgh = w, h
        img = img.reshape(w, h)

    return img

# randomly transform images given the path 
# to the image
def randomtransform(name, size=105):
    img = image.open(name)
    # img = img.resize((size, size))
    a,b,c,d,e,f,g,h = rd.choice(15, size=8)
    angle = rd.choice(35) - 18
    center = tuple(rd.choice(106, size=2))
    coeffs = findcoeffs(
            [(a, b), (105-c, d), (105-e, 105-f), (g, 105-h)],
            [(0, 0), (105, 0), (105, 105), (0, 105)])
    img = img.convert("L")
    img = ops.invert(img)
    if rd.random() < 0.8:
        img = img.rotate(angle=angle, center=center, resample=image.BILINEAR)
    if rd.random() < 0.8:
        img = img.transform((105, 105), image.PERSPECTIVE, data=coeffs, resample=image.BILINEAR)
    if rd.random() < 0.8: 
        data = np.array(img.getdata()).reshape(105, 105)
        data = warp(data, (size, size))
    else:
        img = img.resize((size, size), image.BILINEAR)
        data = np.array(img.getdata()).reshape(size, size)
    img.close() 
    return data  / 255


def loadfiles(path="./omniglot/images_background"):
    walker = os.walk(path)
    path, folders, files = next(walker)
    characters = []
    for file in folders:
        subpath = "{}/{}".format(path, file)
        folders = next(os.walk(subpath))[1]
        for file in folders:
            subsubpath = "{}/{}".format(subpath, file)
            files = [*os.walk(subsubpath)]

            char = []
            characters.append(char)
            names = files[0][2]
            for name in names:
                char.append("{}/{}".format(subsubpath, name))
    return characters

# reads the image and return it without doing anything
def read(path, size=105):
    img = image.open(path)
    img = img.resize((size, size), image.BILINEAR)
    data = np.array(img.getdata()).reshape(size, size)
    return 1.0 - data / 255



characters = loadfiles(path="./omniglot/images_background")
evaluations = loadfiles(path="./omniglot/images_evaluation")
characters = np.array(characters)
evaluations = np.array(evaluations)

# sample from omniglot
def sample(k, mode="training", size=105):
    data = evaluations if mode=="training" else characters
    if mode == "training":
        i = rd.choice(len(data), size=k, replace=False)
        j, k = zip(*[rd.choice(20, size=[2], replace=False) for i in range(k)])
        aset = [randomtransform(s, size) for s in data[i, j]]
        bset = [randomtransform(s, size) for s in data[i, k]]
        return np.array(aset), np.array(bset)
    elif mode == "validating":
        i = rd.choice(100, size=k, replace=False)
        j, k = zip(*[rd.choice(20, size=[2], replace=False) for i in range(k)])
        aset = [read(s, size) for s in data[i, j]]
        bset = [read(s, size) for s in data[i, k]]
        return np.array(aset), np.array(bset)
    elif mode == "testing":
        i = rd.choice(len(data)-100, size=k, replace=False) + 100
        j, k = zip(*[rd.choice(20, size=[2], replace=False) for i in range(k)])
        aset = [read(s, size) for s in data[i, j]]
        bset = [read(s, size) for s in data[i, k]]
        return np.array(aset), np.array(bset)
    else:
        raise ValueError("mode given not known: {}".format(mode))
