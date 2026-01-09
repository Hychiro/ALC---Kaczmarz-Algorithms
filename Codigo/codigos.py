import numpy as np
import matplotlib.pyplot as plt

def solve_kaczmarz(A, b, TOL=1e-6, randomize=False,x0=None,Max = None):
    m, n = A.shape

    if x0 is None:
        X = np.zeros(n)
    else:
        X = x0.reshape(n).copy()

    def calc_err(X):
        return np.linalg.norm(A @ X - b)**2
    b = b.reshape(b.shape[0])
    Xhistory = [X]
    err_history = [calc_err(X)]
    k = 0

    if randomize:
        p = (np.linalg.norm(A, axis=1) / np.linalg.norm(A))**2
    else:
        p = range(m)

    while True:
        if randomize:
            i = np.random.choice(range(m), p=p)
        else:
            i = k % m

        ai = A[i,:]
        Xnew = X + (b[i] - ai @ X) / np.linalg.norm(ai)**2 * ai
        err = calc_err(Xnew)

        Xhistory.append(Xnew)
        err_history.append(err)
        X = Xnew
        if err < TOL:
            break
        if Max is not None and k >= Max*m:
            break
        k += 1

    return X, np.array(Xhistory), np.array(err_history)


def solve_reblock(A, b, k_block=10, lam=1e-3,
                  N=200, Tb=None, TOL=1e-6, x0=None):

    m, n = A.shape
    b = b.reshape(-1)

    def calc_err(X):
        return np.linalg.norm(A @ X - b)**2

    # ----- inicialização com x0 -----
    if x0 is None:
        X = np.zeros(n)
    else:
        X = x0.reshape(n).copy()

    Xhistory = [X.copy()]
    err_history = [calc_err(X)]

    if Tb is not None:
        Xsum = np.zeros_like(X)
        cnt = 0

    if k_block > m:
            k_block = m

 
    p = (np.linalg.norm(A, axis=1) / np.linalg.norm(A))**2
    # número de blocos t
    t = m // k_block if k_block < m else 1
    T = N * t  # total de iterações

    for k in range(T):

        

        S = np.random.choice(m, size=k_block, replace=False, p=p)

        As = A[S, :]
        bs = b[S]

        G = As @ As.T + lam * k_block * np.eye(k_block)
        M = np.linalg.inv(G)

        r = bs - As @ X
        Xnew = X + As.T @ (M @ r)

        err = calc_err(Xnew)
        Xhistory.append(Xnew.copy())
        err_history.append(err)

        if err < TOL:
            X = Xnew
            break

        X = Xnew
        if Tb is not None and k >= Tb:
            Xsum += X
            cnt += 1

    if Tb is None or cnt == 0:
        return X, np.array(Xhistory), np.array(err_history)

    Xbar = Xsum / cnt
    return Xbar, np.array(Xhistory), np.array(err_history)


def solve_BK(A, b, k_block=10, N=100, TOL=1e-6, x0=None):
    """
    Método Block Kaczmarz com blocos de tamanho fixo.
    """
    m, n = A.shape
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.reshape(n).copy()
    b = b.reshape(-1)

    def calc_err(X):
        return np.linalg.norm(A @ X - b)**2

    # Construção dos blocos: cada bloco com 'k_block' linhas
    indices = np.arange(m)
    n_blocks = max(1, m // k_block)  # número de blocos
    blocks = np.array_split(indices, n_blocks)

    t = len(blocks)  # quantidade de blocos

    Xhistory = [x.copy()]
    err_history = [calc_err(x)]

    for k in range(N * t):
        tau_k = blocks[k % t]  # determinístico
        A_tau = A[tau_k, :]
        b_tau = b[tau_k].ravel()

        x = x + np.linalg.pinv(A_tau) @ (b_tau - A_tau @ x)

        err = calc_err(x)
        Xhistory.append(x.copy())
        err_history.append(err)

        if err <= TOL:
            return x, np.array(Xhistory), np.array(err_history)

    return x, np.array(Xhistory), np.array(err_history)




def solve_RBK_U(A, b, TOL=1e-6, randomize=True,
                x0=None, N=None, k_block=10, Tb=None):
    m, n = A.shape

    # Inicialização
    if x0 is None:
        X = np.zeros(n)
    else:
        X = x0.reshape(n).copy()
    b = b.reshape(b.shape[0])
    def calc_err(X):
        return np.linalg.norm(A @ X - b)**2

    Xhistory = [X.copy()]
    err_history = [calc_err(X)]
    k = 0
    if k_block > m:
        k_block = m

    t = m // k_block if k_block < m else 1
    T = N * t  # total de iterações

    p = (np.linalg.norm(A, axis=1) / np.linalg.norm(A))**2

    for k in range(T):

        # Amostragem do bloco S_t ~ U(m, k)
        if randomize:
            S = np.random.choice(m, size=k_block, replace=False,p=p)
        else:
            start = (k * k_block) % m
            S = np.arange(start, min(start + k_block, m))

        A_S = A[S, :]
        b_S = b[S]

        # M(A_S) = (A_S A_S^T)^+
        G = A_S @ A_S.T
        M = np.linalg.pinv(G)

        # Atualização RBK-U
        Xnew = X + A_S.T @ M @ (b_S - A_S @ X)

        err = calc_err(Xnew)

        Xhistory.append(Xnew.copy())
        err_history.append(err)
        X = Xnew

        # Critério de parada
        if err < TOL:
            break

    Xhistory = np.array(Xhistory)
    err_history = np.array(err_history)

    # Tail averaging (opcional)
    if Tb is not None and Tb < len(Xhistory) - 1:
        Xbar = np.mean(Xhistory[Tb+1:], axis=0)
        return Xbar, Xhistory, err_history

    return X, Xhistory, err_history