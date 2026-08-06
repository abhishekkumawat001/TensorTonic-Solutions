def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    # No history -> steepest descent
    if len(s_list) == 0:
        return [-g for g in grad]

    m = len(s_list)

    # rho_i = 1 / (y_i^T s_i)
    rho = [1.0 / _dot(y_list[i], s_list[i]) for i in range(m)]

    # ---------- First Loop (backward) ----------
    q = grad[:]
    alpha = [0.0] * m

    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * _dot(s_list[i], q)
        q = [
            qj - alpha[i] * yj
            for qj, yj in zip(q, y_list[i])
        ]

    # ---------- Initial Hessian Scaling ----------
    s_last = s_list[-1]
    y_last = y_list[-1]

    gamma = _dot(s_last, y_last) / _dot(y_last, y_last)

    r = [gamma * qj for qj in q]

    # ---------- Second Loop (forward) ----------
    for i in range(m):
        beta = rho[i] * _dot(y_list[i], r)
        r = [
            rj + sj * (alpha[i] - beta)
            for rj, sj in zip(r, s_list[i])
        ]

    # Descent direction
    return [-x for x in r]