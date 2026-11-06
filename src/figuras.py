"""As três figuras do README, geradas do mesmo cálculo das saídas de texto.

    uv run python src/figuras.py   -> docs/figuras/*.png  (e o resumo em outputs/10-figuras.txt)

matplotlib com backend Agg: sem janela, roda em servidor. Cada figura imprime os numeros
que desenha, para o texto do README poder ser conferido contra `outputs/` sem abrir PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import metricas  # noqa: E402
import prepare  # noqa: E402
from camada_1_erro_e_subgrupos import wilson  # noqa: E402
from modelo_e_baselines import logistica  # noqa: E402
from prepare import CATEGORICAS  # noqa: E402

DESTINO = Path(__file__).resolve().parents[1] / "docs" / "figuras"
RAZAO = metricas.RAZAO_REFERENCIA

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.hashsalt": "fixo", "path.simplify": True})


def salvar(fig, nome: str) -> None:
    fig.tight_layout()
    fig.savefig(DESTINO / nome, metadata={"Software": None})
    plt.close(fig)
    print(f"   gravado: docs/figuras/{nome}")


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    partes = prepare.conjuntos()
    X_tr, y_tr = prepare.X_y(partes["treino"])
    X_cal, y_cal = prepare.X_y(partes["calibracao"])
    teste = partes["teste"]
    X_te, y_te = prepare.X_y(teste)
    y_te = np.asarray(y_te)

    modelo = logistica([c for c in X_tr.columns if c not in CATEGORICAS]).fit(X_tr, y_tr)
    p_cal = modelo.predict_proba(X_cal)[:, 1]
    p_te = modelo.predict_proba(X_te)[:, 1]
    t, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(RAZAO))

    # 1. Curva de custo no teste, com o limiar escolhido na calibracao.
    limiares = np.linspace(0.005, 0.6, 240)
    custos = np.array([metricas.custo(y_te, p_te, x, 1.0, float(RAZAO)) for x in limiares])
    c_t = metricas.custo(y_te, p_te, t, 1.0, float(RAZAO))
    c_50 = metricas.custo(y_te, p_te, 0.5, 1.0, float(RAZAO))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(limiares, custos, color="#1F5D45")
    ax.axvline(t, color="#1F5D45", ls="--", lw=1)
    ax.axvline(0.5, color="#9E3925", ls=":", lw=1)
    ax.annotate(f"limiar escolhido na calibração\nt = {t:.3f}  custo {c_t:,.0f}",
                (t, c_t), xytext=(t + 0.05, c_t + 0.25 * (custos.max() - c_t)),
                arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=9)
    ax.annotate(f"padrão 0,5\ncusto {c_50:,.0f}", (0.5, c_50), xytext=(0.36, c_50 * 1.12),
                arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=9)
    ax.set_xlabel("limiar de decisão (probabilidade prevista de calote)")
    ax.set_ylabel(f"custo no teste (1 FN = {RAZAO} FP)")
    ax.set_title("Custo de operar em cada limiar · dado sintético", loc="left", fontsize=11)
    salvar(fig, "custo_limiar.png")
    print(f"      t* = {t:.4f} | custo em t* {c_t:,.0f} | custo em 0,5 {c_50:,.0f} "
          f"({c_50 / c_t:.2f}x)")

    # 2. Recall por regiao com IC 95% de Wilson.
    geral = metricas.matriz(y_te, p_te, t)
    regioes = sorted(teste["regiao"].unique())
    linhas = []
    for g in regioes:
        sel = (teste["regiao"] == g).to_numpy()
        r = metricas.matriz(y_te[sel], p_te[sel], t)
        lo, hi = wilson(r["vp"], r["vp"] + r["fn"])
        linhas.append((g, r["recall"], lo, hi, r["vp"] + r["fn"]))
    linhas.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (g, rec, lo, hi, pos) in enumerate(linhas):
        cor = "#9E3925" if hi < geral["recall"] else "#1F5D45"
        ax.errorbar(rec, i, xerr=[[rec - lo], [hi - rec]], fmt="o", color=cor, capsize=4)
        ax.text(1.02, i, f"{rec:.2f} [{lo:.2f}; {hi:.2f}]  n+={pos}", va="center", fontsize=8.5,
                transform=ax.get_yaxis_transform())
    ax.axvline(geral["recall"], color="#47594F", ls="--", lw=1)
    ax.text(geral["recall"], len(linhas) - 0.45, f" geral {geral['recall']:.2f}", fontsize=9,
            color="#47594F")
    ax.set_yticks(range(len(linhas)), [x[0] for x in linhas])
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall: fração dos calotes retidos (IC 95%, Wilson)")
    ax.set_title("Recall por região no limiar 10:1 · Norte plantado no gerador", loc="left",
                 fontsize=11)
    salvar(fig, "recall_por_regiao.png")
    for g, rec, lo, hi, pos in linhas:
        print(f"      {g:<14} recall {rec:.3f} [{lo:.3f} ; {hi:.3f}]  positivos {pos}")

    # 3. Diagrama de confiabilidade (bins de mesma populacao).
    bordas = np.unique(np.quantile(p_te, np.linspace(0, 1, 9)))
    idx = np.clip(np.digitize(p_te, bordas[1:-1]), 0, len(bordas) - 2)
    prev = [p_te[idx == b].mean() for b in range(len(bordas) - 1)]
    obs = [y_te[idx == b].mean() for b in range(len(bordas) - 1)]
    los, his = zip(*[wilson(int(y_te[idx == b].sum()), int((idx == b).sum()))
                     for b in range(len(bordas) - 1)])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    lim = max(max(prev), max(his)) * 1.08
    ax.plot([0, lim], [0, lim], color="#999", lw=1, ls="--", label="calibração perfeita")
    ax.errorbar(prev, obs, yerr=[np.array(obs) - los, np.array(his) - obs], fmt="o-",
                color="#1F5D45", capsize=3, label="logística, sem recalibrar")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("probabilidade prevista (média da faixa)")
    ax.set_ylabel("taxa observada de calote")
    L = metricas.linha("", y_te, p_te)
    ax.set_title(f"Calibração no teste · ECE {L['ece']:.3f}", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    salvar(fig, "calibracao.png")
    print(f"      ECE {L['ece']:.4f} | Brier {L['brier']:.5f} | maior gap entre previsto e "
          f"observado {max(abs(a - b) for a, b in zip(prev, obs)):.3f}")


if __name__ == "__main__":
    main()
