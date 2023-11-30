import numpy
import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)

def h_more_struct_1(z):
    num_vals = 16
    b = num_vals * (num_vals - 1) // 2

    eigvals = z[:num_vals]
    eigvec_product_R = np.zeros((num_vals, num_vals))
    eigvec_product_I = np.zeros((num_vals, num_vals))

    eigvec_product_R = eigvec_product_R.at[np.triu_indices_from(eigvec_product_R, k=1)].set(z[num_vals:num_vals+b])
    eigvec_product_I = eigvec_product_I.at[np.triu_indices_from(eigvec_product_I, k=1)].set(z[num_vals+b:])

    #count = -1
    #for i in range(num_vals):
    #    for j in range(i + 1, num_vals):
    #        count += 1
    #        eigvec_product_R = eigvec_product_R.at[i, j].set(z[num_vals + count])
    #        eigvec_product_I = eigvec_product_I.at[i, j].set(z[num_vals + b + count])

    mineig = np.min(eigvals)
    #tol = np.where(mineig > 1e-8, mineig, 1e-8)

    running_sum = 0
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            numer = (eigvals[i] - eigvals[j]) ** 2
            term = eigvec_product_R[i, j] ** 2 + eigvec_product_I[i, j] ** 2
            running_sum += (numer / denom * term) #*np.where(denom*denom > tol*tol, 1, 0)
    return 4 * running_sum

@jax.jit
def ad_jvp_jit(z, zd):
    resd = jax.jvp(h_more_struct_1, (z,) , (zd,))
    return resd

@jax.jit
def ad_jvpjvp_jit(z, zd, zdt, zdd):
    _, resdd = jax.jvp(ad_jvp_jit, (z, zd) , (zdt, zdd))
    return resdd

def ad_jacobian(Cres, Gres):
    n, m = Gres.shape
    G = numpy.zeros(n)
    for i in range(n):
        _, G[i] = ad_jvp_jit(np.array(Cres), np.array(Gres[i, :]))
    return G

def ad_hessian(Cres, Gres, Hres):
    n, _, m = Hres.shape
    H = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            _, H[i, j] = ad_jvpjvp_jit(np.array(Cres), np.array(Gres[i,:]), np.array(Gres[j,:]), np.array(Hres[i, j, :]))
    return H

#for i in range(10):
for i in range(1):
    #numpy.random.seed(i)
    #m = 256 # input dim
    #n = numpy.random.randint(2, 20)   # output dim
    #Cres = numpy.random.uniform(-1, 1, m)
    #Gres = numpy.random.uniform(-1, 1, (n, m))
    #Hres = numpy.zeros((n, n, m))
    #for j in range(m):
    #    Hressqrt = numpy.random.rand(n, n)
    #    Hres[:, :, j] = Hressqrt @ Hressqrt.T

    B = np.load('ex3_z.npy', allow_pickle=True).flatten()[0]
    Cres = B['Cres']
    Gres = B['Gres']
    Hres = B['Hres']

    G_ad = ad_jacobian(Cres, Gres)
    print("G_ad: ", (G_ad))
    H_ad = ad_hessian(Cres, Gres, Hres)
    print("H_ad: ", (H_ad))
