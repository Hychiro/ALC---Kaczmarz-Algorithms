import numpy as np
import matplotlib.pyplot as plt
import time
from Codigo.codigos import *
def run_pipeline(A, b, x0, params):

    results = {}

    print("1) KACZMARZ DETERMINÍSTICO")
    start = time.time()
    # ---------- 1) KACZMARZ DETERMINÍSTICO ----------
    X_det, Xh_det, err_det = solve_kaczmarz(
        A, b,

        TOL=params["epsilon"],
        randomize=False,
        Max = params["N_single_row"],
    )
    print("erro final =", err_det[-1])
    print("erro minimio =", np.min(err_det))
    results["kaczmarz_deterministic"] = {
        "X": X_det,
        "err": err_det,
        "iters": len(err_det) - 1
    }
    end = time.time()
    print("Tempo de execução:", end - start)
    print("2) KACZMARZ ALEATÓRIO (média)")
    start = time.time()

    # ---------- 2) KACZMARZ ALEATÓRIO (média) ----------
    runs = params["runs_randomized"]
    all_errs = []
    X_last = None

    for _ in range(runs):
        X_rk, Xh_rk, err_rk = solve_kaczmarz(
            A, b,

            TOL=params["epsilon"],
            randomize=True,
            Max = params["N_single_row"],
        )
        all_errs.append(np.asarray(err_rk))
        X_last = X_rk

    # --- métricas por execução (sem truncagem) ---
    final_errs = [errs[-1] for errs in all_errs]   # erro final de cada run
    min_errs   = [errs.min() for errs in all_errs] # menor erro atingido em cada run

    mean_final_err = np.mean(final_errs)
    mean_min_err   = np.mean(min_errs)

    print("erro final médio =", mean_final_err)
    print("erro mínimo médio =", mean_min_err)

    # runs = len(all_errs)
    max_len = max(len(errs) for errs in all_errs)

    aligned_errs = np.empty((runs, max_len))

    for i, errs in enumerate(all_errs):
        length = len(errs)
        aligned_errs[i, :length] = errs
        # Repete o último valor até completar o comprimento máximo
        aligned_errs[i, length:] = errs[-1]


    # --- salvar resultados ---
    results["kaczmarz_randomized"] = {
        "X": X_last,
        "err_mean": np.nanmean(aligned_errs, axis=0),
        "err_std":  np.nanstd(aligned_errs, axis=0),
        "runs": runs,
        "final_errs": final_errs,
        "min_errs": min_errs,
        "mean_final_err": mean_final_err,
        "mean_min_err": mean_min_err,
    }


    end = time.time()
    print("Tempo de execução:", end - start)
    print("3) ReBlocK")
    start = time.time()
    # ---------- 3) ReBlocK ----------
    runs = 10  # número de execuções para média
    all_errs_rb = []
    x_last_rb = None

    for _ in range(runs):
        X_rb, Xh_rb, err_rb = solve_reblock(
            A, b,
            k_block=params["block_size"],
            lam=0.001,
            N=params["N_block"],
            Tb=None,
            TOL=params["epsilon"],
        )
        all_errs_rb.append(np.asarray(err_rb))
        x_last_rb = X_rb

    # métricas por execução
    final_errs_rb = [errs[-1] for errs in all_errs_rb]
    min_errs_rb   = [errs.min() for errs in all_errs_rb]

    mean_final_err_rb = np.mean(final_errs_rb)
    mean_min_err_rb   = np.mean(min_errs_rb)

    print("erro final médio =", mean_final_err_rb)
    print("erro mínimo médio =", mean_min_err_rb)

    # alinhar comprimentos
    max_len_rb = max(len(errs) for errs in all_errs_rb)
    aligned_errs_rb = np.empty((runs, max_len_rb))

    for i, errs in enumerate(all_errs_rb):
        length = len(errs)
        aligned_errs_rb[i, :length] = errs
        aligned_errs_rb[i, length:] = errs[-1]  # padding com último valor

    results["reblock"] = {
        "X": x_last_rb,
        "err_mean": np.nanmean(aligned_errs_rb, axis=0),
        "err_std":  np.nanstd(aligned_errs_rb, axis=0),
        "runs": runs,
        "final_errs": final_errs_rb,
        "min_errs": min_errs_rb,
        "mean_final_err": mean_final_err_rb,
        "mean_min_err": mean_min_err_rb,
    }

    end = time.time()
    print("Tempo de execução:", end - start)

    print("4) BLOCK KACZMARZ")
    start = time.time()
    # ---------- 4) BLOCK KACZMARZ ----------
    x_block, x_hist_block, err_hist_block = solve_BK(
        A, b,
        k_block=params["block_size"],
        N=params["N_block"],
        TOL=params["epsilon"],
    )
    print("erro final =", err_hist_block[-1])
    print("erro mínimo =", np.min(err_hist_block))
    results["BK"] = {
        "X": x_block,
        "err": err_hist_block,
        "iters": len(err_hist_block) - 1
    }
    end = time.time()
    print("Tempo de execução:", end - start) 
    print("5) RANDOM BLOCK KACZMARZ")
    start = time.time()
    # ---------- 5) RANDOM BLOCK KACZMARZ ----------
    runs = 10  # número de execuções para média
    all_errs = []
    x_last_rbk = None

    for _ in range(runs):
        x_rbk, x_hist_rbk, err_hist_rbk = solve_RBK_U(
            A, b,
            k_block=params["block_size"],
            N=params["N_block"],
            TOL=params["epsilon"],
        )
        all_errs.append(np.asarray(err_hist_rbk))
        x_last_rbk = x_rbk

    # métricas por execução
    final_errs = [errs[-1] for errs in all_errs]
    min_errs   = [errs.min() for errs in all_errs]

    mean_final_err = np.mean(final_errs)
    mean_min_err   = np.mean(min_errs)

    print("erro final médio =", mean_final_err)
    print("erro mínimo médio =", mean_min_err)

    # alinhar comprimentos
    max_len = max(len(errs) for errs in all_errs)
    aligned_errs = np.empty((runs, max_len))

    for i, errs in enumerate(all_errs):
        length = len(errs)
        aligned_errs[i, :length] = errs
        aligned_errs[i, length:] = errs[-1]  # padding com último valor

    results["RBK"] = {
        "X": x_last_rbk,
        "err_mean": np.nanmean(aligned_errs, axis=0),
        "err_std":  np.nanstd(aligned_errs, axis=0),
        "runs": runs,
        "final_errs": final_errs,
        "min_errs": min_errs,
        "mean_final_err": mean_final_err,
        "mean_min_err": mean_min_err,
    }
    end = time.time()
    print("Tempo de execução:", end - start)



    return results

"""## Plots"""

def plot_convergence(results, fname):

    plt.figure(figsize=(8,5))

    # --- Kaczmarz determinístico ---
    err = results["kaczmarz_deterministic"]["err"]
    plt.plot(err, '--', label="Original Kaczmarz")

    # --- Kaczmarz aleatório (média) ---
    err_mean = results["kaczmarz_randomized"]["err_mean"]
    k = np.arange(len(err_mean))
    plt.plot(k, err_mean, '--', label="Randomized Kaczmarz (média)")
    # opcional: faixa de desvio padrão

    # --- ReBlocK ---
    err_reblock = results["reblock"]["err_mean"]
    k_reblock = np.arange(len(err_reblock))
    plt.plot(k_reblock, err_reblock, '--', label="ReBlocK")

    # --- Block Kaczmarz (BK) ---
    err_bk = results["BK"]["err"]
    plt.plot(err_bk, '--', label="Block Kaczmarz")

    # --- Random Block Kaczmarz (RBK) ---
    # se você salvou como média de runs

    err_mean_rbk = results["RBK"]["err_mean"]
    k_rbk = np.arange(len(err_mean_rbk))
    plt.plot(k_rbk, err_mean_rbk, '--', label="Random Block Kaczmarz (média)")

    # estilo
    plt.yscale("log")
    plt.xlabel("Iterações")
    plt.ylabel("||Ax - b||")
    plt.title("Convergência dos métodos Kaczmarz")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"convergence_kaczmarz_{fname.replace('.npz','')}.png",
                dpi=200, bbox_inches="tight")
    plt.show()

arquivos = {
    "5.1.2  (underdetermined)": "dados_experimento_5_1_2.npz",
    "5.1.3  (overdetermined cond=1e5)": "dados_experimento_5_1_3.npz",
    "5.1.5  (overdetermined cond≈1.1)": "dados_experimento_5_1_5.npz"
}

for titulo, fname in arquivos.items():
    data = np.load(fname, allow_pickle=True)
    A = data["A"]
    x = data["x"]
    b = data["b"]
    x0 = data["x0"]
    params = data["params"].item()
    print(f"Rodando {titulo}")
    results = run_pipeline(A, b,x0, params)
    plot_convergence(results, fname)