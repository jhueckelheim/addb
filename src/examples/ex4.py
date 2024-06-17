import numpy
import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)

def norm2(z):
    res = 1 - (1 - np.prod(z[:-1]))/(1 - np.prod(z[-1]*z[:-1]))
    return res

def norm2_fd(z, zd, h):
    resd = norm2(z+h*zd)-norm2(z-h*zd)
    resd = resd/2*h

    return resd

def norm2_fdd(z, zd, h):
    resdd = norm2(z+h*zd) + norm2(z-h*zd) - 2*norm2(z)
    resdd = resdd/(h**2)

    return resdd

@jax.jit
def norm2_d(z, zd):
    resd = jax.jvp(norm2, (z,) , (zd,))
    return resd

@jax.jit
def norm2_dd(z, zd, zdt, zdd):
    _, resdd = jax.jvp(norm2_d, (z, zd) , (zdt, zdd))
    return resdd

def G_norm2(Cres, Gres):
    n, m = Gres.shape
    G = numpy.zeros(n)
    for i in range(n):
        _, G[i] = norm2_d(Cres, Gres[i, :])
    return G

def H_norm2(Cres, Gres, Hres):
    n, _, m = Hres.shape
    H = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            _, H[i, j] = norm2_dd(Cres, Gres[i,:], Gres[j,:], Hres[i, j, :])
    return H

def G_norm2_fd(Cres, Gres):
    n, m = Gres.shape
    G = numpy.zeros(n)
    for i in range(n):
        _, G[i] = norm2_fd(Cres, Gres[i, :])
    return G

def H_norm2_fd(Cres, Gres, Hres):
    n, _, m = Hres.shape
    H = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            _, H[i, j] = norm2_fdd(Cres, Gres[i,:], Gres[j,:], Hres[i, j, :])
    return H

# # F = Cres(x) * Cres(x)
# # G = 2*Cres(x) * Gres(x)
# # H = 2*Gres(x) * Hres + Cres(x)*Hres(x)
# def leastsquares(Cres, Gres, Hres):
#     n, _, m = Hres.shape

#     G = 2 * Gres @ Cres.T
#     H = np.zeros((n, n))
#     for i in range(m):
#         H = H + Cres[i] * Hres[:, :, i]

#     H = 2 * H + 2 * Gres @ Gres.T

#     return G, H

#for i in range(10):
for i in range(1):
    numpy.random.seed(i)
    m = np.array(numpy.random.randint(30, 31)) # input dim
    n = np.array(numpy.random.randint(4, 5))   # output dim
    Cres = np.array(numpy.random.uniform(-1, 1, m))
    Gres = np.array(numpy.random.uniform(-1, 1, (n, m)))

    Hres = numpy.zeros((n, n, m))
    for j in range(m):
        Hressqrt = numpy.random.rand(n, n)
        Hres[:, :, j] = Hressqrt @ Hressqrt.T
    Hres = np.array(Hres)

    G_fd = norm2_fd(Cres)
    H_fd = norm2_fdd(

    G_ad = G_norm2(Cres, Gres)
    print("G_ad - G: ", (G_ad - G))
    H_ad = H_norm2(Cres, Gres, Hres)
    print("H_ad - H: ", (H_ad - H))
    print(G_ad, H_ad)
