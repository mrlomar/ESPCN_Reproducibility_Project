import numpy as np
from math import floor

def PS(T, r):
    rW = r*len(T)
    rH = r*len(T[0])
    C = len(T[0][0])/(r*r)

    # make sure C is an integer and cast if this is the case
    assert(C == int(C))
    C = int(C)
    
    res = np.zeros((rW, rH, C))

    for x in range(len(res)):
        for y in range(len(res[x])):
            for c in range(len(res[x][y])):
                res[x][y][c] = \
                    T[floor(x/r)][floor(y/r)][C*r*(y % r) + C*(x % r) + c]
    return res

def PS_inv(img, r):
    r2 = r*r
    W = len(img)/r
    H = len(img[0])/r
    C = len(img[0][0])
    Cr2 = C*r2

    # Make sure H and W are integers
    assert(int(H) == H and int(W) == W)
    H, W = int(H), int(W)

    res = np.zeros((W, H, Cr2))

    for x in range(len(img)):
        for y in range(len(img[x])):
            for c in range(len(img[x][y])):
                res[floor(x/r)][floor(y/r)][C*r*(y % r) + C*(x % r) + c] = img[x][y][c]
    return res

def shuffle_loss(output, target, r=4):
	# Simply a custom loss function with the periodic shuffle built in
	# Using this allows us to read in the images without reverse shuffling for training
	# Might be slower because of the calculations, but still interesting to try
	# The paper does not specify what to do with the colour channels
	res = 0
	rW = len(target)
	rH = len(target[0])
	C = len(target[0][0])

	for x in range(rW):
		for y in range(rH):
			for c in range(C):
				res += (target[x][y][c] - output[floor(x/r)][floor(y/r)][C*r*(y % r) + C*(x % r) + c])**2

	return res / (rW*rH*c)

random_img = np.random.rand(16, 16, 3)
exp_output = PS_inv(random_img, 4)
print(exp_output.shape)
print(shuffle_loss(exp_output, random_img))

noisy_output = exp_output + np.random.rand(4, 4, 48)*0.01
print(shuffle_loss(noisy_output, random_img))

