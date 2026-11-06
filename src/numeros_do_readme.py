"""Todo número citado no README, num arquivo só, para conferir prosa contra saída.

Número em prosa envelhece calado. Este script recalcula cada número que o README cita e
grava em `outputs/11-numeros-do-readme.txt`; a tabela "de onde vem cada número", no
README, aponta para cá e para as saídas detalhadas.

    uv run python src/numeros_do_readme.py
"""
from __future__ import annotations

import numpy as np

import metricas
import prepare
from camada_1_erro_e_subgrupos import wilson
from modelo_e_baselines import boosting, floresta, logistica
from prepare import ALVO, CATEGORICAS


def main() -> None:
    bruto = prepare.bruto()
    partes = prepare.conjuntos()
    X_tr, y_tr = prepare.X_y(partes["treino"])
    X_cal, y_cal = prepare.X_y(partes["calibracao"])
    teste = partes["teste"]
    X_te, y_te = prepare.X_y(teste)
    y_te = np.asarray(y_te)
    num = [c for c in X_tr.columns if c not in CATEGORICAS]

    print("NUMEROS CITADOS NO README (dado sintetico)\n")
    print(f"[dado] linhas {len(bruto):,} | prevalencia {bruto[ALVO].mean():.2%} | "
          f"teste {len(y_te):,} com {int(y_te.sum())} inadimplentes")
    for g, taxa in bruto.groupby("regiao")[ALVO].mean().items():
        print(f"[dado] taxa de calote {g:<13} {taxa:.2%}")

    print("\n[modelos] AUPRC na calibracao (piso = prevalencia "
          f"{y_cal.mean():.4f})")
    regra = metricas.linha("", y_cal, (850 - X_cal["score_bureau"]) / 550)["auprc"]
    print(f"   regra do score        {regra:.4f}  ({regra / y_cal.mean():.1f}x o piso)")
    auprc = {}
    for nome, f in [("logistica", logistica), ("floresta", floresta), ("boosting", boosting)]:
        m = f(num).fit(X_tr, y_tr)
        auprc[nome] = metricas.linha("", y_cal, m.predict_proba(X_cal)[:, 1])["auprc"]
        treino = metricas.linha("", y_tr, m.predict_proba(X_tr)[:, 1])["auprc"]
        print(f"   {nome:<21} {auprc[nome]:.4f}  ({auprc[nome] / y_cal.mean():.1f}x o piso)"
              f"   no proprio treino {treino:.4f}")
    print(f"   ganho da logistica sobre a regra: {auprc['logistica'] - regra:+.4f}")

    modelo = logistica(num).fit(X_tr, y_tr)
    p_cal = modelo.predict_proba(X_cal)[:, 1]
    p_te = modelo.predict_proba(X_te)[:, 1]
    t, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(metricas.RAZAO_REFERENCIA))
    m = metricas.matriz(y_te, p_te, t)
    m50 = metricas.matriz(y_te, p_te, 0.5)
    L = metricas.linha("", y_te, p_te)
    print(f"\n[teste] AUPRC {L['auprc']:.4f} (piso {y_te.mean():.4f}) | AUC {L['auc']:.4f} | "
          f"ECE {L['ece']:.4f} | Brier {L['brier']:.5f} (piso {y_te.mean() * (1 - y_te.mean()):.5f})")
    print(f"[limiar] razao {metricas.RAZAO_REFERENCIA}:1 -> t* {t:.4f} (escolhido na calibracao)")
    print(f"[limiar] em t*: VP {m['vp']} FN {m['fn']} FP {m['fp']} VN {m['vn']} | recall "
          f"{m['recall']:.3f} | precisao {m['precisao']:.3f} | negados {m['vp'] + m['fp']:,} "
          f"({(m['vp'] + m['fp']) / len(y_te):.1%})")
    print(f"[limiar] em 0,5: recall {m50['recall']:.3f} ({m50['vp']} de {int(y_te.sum())})")
    c_t = metricas.custo(y_te, p_te, t, 1, metricas.RAZAO_REFERENCIA)
    c_50 = metricas.custo(y_te, p_te, 0.5, 1, metricas.RAZAO_REFERENCIA)
    print(f"[limiar] custo em t* {c_t:,.0f} | em 0,5 {c_50:,.0f} | razao {c_50 / c_t:.2f}x")

    print("\n[subgrupos] recall por regiao no limiar de referencia (IC 95% Wilson)")
    for g in sorted(teste["regiao"].unique()):
        sel = (teste["regiao"] == g).to_numpy()
        r = metricas.matriz(y_te[sel], p_te[sel], t)
        lo, hi = wilson(r["vp"], r["vp"] + r["fn"])
        print(f"   {g:<14} {r['recall']:.3f} [{lo:.3f} ; {hi:.3f}]  positivos "
              f"{r['vp'] + r['fn']}  {'PIOR' if hi < m['recall'] else ''}")
    lo, hi = wilson(m["vp"], m["vp"] + m["fn"])
    print(f"   {'(geral)':<14} {m['recall']:.3f} [{lo:.3f} ; {hi:.3f}]")

    sel = (teste["regiao"] == "Norte").to_numpy()
    reais = 0
    for razao in metricas.FAIXA_RAZOES:
        tr, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(razao))
        g = metricas.matriz(y_te, p_te, tr)
        s = metricas.matriz(y_te[sel], p_te[sel], tr)
        reais += wilson(s["vp"], s["vp"] + s["fn"])[1] < g["recall"]
    print(f"[sensibilidade] Norte pior com IC fora do geral em {reais} de "
          f"{len(metricas.FAIXA_RAZOES)} cenarios de custo")

    ordem = np.argsort(np.abs(p_te - t))
    k = int(round(len(y_te) * 0.10))
    fica = np.ones(len(y_te), bool)
    fica[ordem[:k]] = False
    m90 = metricas.matriz(y_te[fica], p_te[fica], t)
    print(f"\n[abstencao] cobertura 90%: {k} a revisao | precisao {m['precisao']:.3f} -> "
          f"{m90['precisao']:.3f} | recall {m['recall']:.3f} -> {m90['recall']:.3f}")


if __name__ == "__main__":
    main()
