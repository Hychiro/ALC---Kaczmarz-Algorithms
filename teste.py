import numpy as np
import matplotlib.pyplot as plt
import time
from Codigo.codigosTestes import *
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

    # --- alinhar comprimentos com NaN padding ---
    # max_len = max(len(errs) for errs in all_errs)
    # aligned_errs = np.full((runs, max_len), np.nan)

    # for i, errs in enumerate(all_errs):
    #     aligned_errs[i, :len(errs)] = errs


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
    print("1) ReBlocK")
    start = time.time()
    # ---------- 3) ReBlocK ----------
    X_rb, Xh_rb, err_rb = solve_reblock(
        A, b,
        k_block=params["block_size"],
        lam=1e-2,
        N=params["N_block"],
        Tb=None,
        TOL=params["epsilon"],
            )
    print("erro final =", err_rb[-1])
    print("erro minimio =", np.min(err_rb))
    results["reblock"] = {
        "X": X_rb,
        "err": err_rb,
        "iters": len(err_rb) - 1
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

    # --- Kaczmarz aleatório (média + desvio) ---
    err_mean = results["kaczmarz_randomized"]["err_mean"]
    err_std  = results["kaczmarz_randomized"]["err_std"]
    mean_final_err = results["kaczmarz_randomized"]["mean_final_err"]
    mean_min_err   = results["kaczmarz_randomized"]["mean_min_err"]

    k = np.arange(len(err_mean))



    # curva média
    plt.plot(k, err_mean, '--', label="Randomized Kaczmarz (média)")

    # --- ReBlocK ---
    err_rb = results["reblock"]["err"]
    plt.plot(err_rb, '--', label="ReBlocK")

    # estilo
    plt.yscale("log")
    plt.xlabel("Iterações")
    plt.ylabel("||Ax - b||")
    plt.title("Convergência do Kaczmarz Aleatório")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"convergence_kaczmarz_{fname.replace('.npz','')}.png",
                        dpi=200, bbox_inches="tight")
    plt.show()

arquivos = {
    # "5.1.2  (underdetermined)": "dados_experimento_5_1_2.npz",
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