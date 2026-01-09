import numpy as np
import matplotlib.pyplot as plt
import time
def solve_kaczmarz(A, b, TOL=1e-6, randomize=False,x0=None,Max = None):
    m, n = A.shape
    start = time.time()
    if x0 is None:
        X = np.zeros(n)
    else:
        X = x0.reshape(n).copy()
    b = b.reshape(b.shape[0])

    def calc_err(X):
        return np.max(np.abs(A @ X - b))


    end = time.time()
    print("Tempo de execução1:", end - start)
    Xhistory = [X]
    err_history = [calc_err(X)]
    k = 0
    start = time.time()
    if randomize:
        p = (np.linalg.norm(A, axis=1) / np.linalg.norm(A))**2
    else:
        p = range(m)
    end = time.time()
    print("Tempo de execução2:", end - start)
    while True:
        start = time.time()
        if randomize:
            i = np.random.choice(range(m), p=p)
        else:
            i = k % m
        end = time.time()
        print(f"Tempo de execução 1 itr {k}:", end - start)
        start = time.time()
        ai = A[i,:]
        end = time.time()
        print(f"Tempo de execução 2 itr {k}:", end - start)
        start = time.time()
        Xnew = X + (b[i] - ai @ X) / np.linalg.norm(ai)**2 * ai
        end = time.time()
        print(f"Tempo de execução 3 itr {k}:", end - start)
        start = time.time()
        err = calc_err(Xnew)
        end = time.time()
        print(f"Tempo de execução 4 itr {k}:", end - start)
        start = time.time()

        Xhistory.append(Xnew)
        end = time.time()
        print(f"Tempo de execução 5 itr {k}:", end - start)
        start = time.time()
        err_history.append(err)
        end = time.time()
        print(f"Tempo de execução 6 itr {k}:", end - start)
        start = time.time()
        X = Xnew
        
        if err < TOL:
            break
        if Max is not None and k >= Max*m:
            break
        k += 1

    return X, np.array(Xhistory), np.array(err_history)


def solve_reblock(A, b, k_block=10, lam=1e-2,
                  N=200, Tb=None, TOL=1e-6, x0=None):

    m, n = A.shape
    b = b.reshape(-1)

    def calc_err(x):
        return np.max(np.abs(A @ x - b))

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

    rng = np.random.default_rng()

    # número de blocos t
    t = m // k_block if k_block < m else 1
    T = N * t  # total de iterações

    for k in range(T):

        if k_block > m:
            k_block = m

        S = rng.choice(m, size=k_block, replace=False)

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