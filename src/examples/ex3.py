def h_more_struct_1(z):
    num_vals = 16
    b = num_vals * (num_vals - 1) // 2

    eigvals = z[:num_vals]
    eigvec_product_R = np.zeros((num_vals, num_vals))
    eigvec_product_I = np.zeros((num_vals, num_vals))

    count = -1
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            count += 1
            eigvec_product_R[i, j] = z[num_vals + count]
            eigvec_product_I[i, j] = z[num_vals + b + count]

    tol = max(1e-8, -10 * np.min(eigvals))

    running_sum = 0
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            if not np.isclose(denom, 0, atol=tol, rtol=tol):
                numer = (eigvals[i] - eigvals[j]) ** 2
                term = eigvec_product_R[i, j] ** 2 + eigvec_product_I[i, j] ** 2
                running_sum += numer / denom * term
    return 4 * running_sum
