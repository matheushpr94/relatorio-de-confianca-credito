"""Que tipo de erro o modelo comete, e com quem ele erra?

Camada 1: a matriz celula por celula, as metricas com o piso ao lado, e o fatiamento,
porque metrica agregada esconde falha localizada.

O limiar de referencia e' o da razao 1 FN = 10 FPs, escolhido na CALIBRACAO. Ninguem do
negocio assinou essa razao: ela esta' aqui como premissa declarada, e a tabela da camada 2
mostra o que muda com as outras.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

import metricas
import prepare
from modelo_e_baselines import logistica
from prepare import ALVO, CATEGORICAS

RAZAO = metricas.RAZAO_REFERENCIA


def titulo(t: str) -> None:
    print(f"\n{'=' * 84}\n{t}\n{'=' * 84}")


def wilson(acertos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de 95% para uma proporcao, pelo metodo de Wilson.

    Wilson, e nao o intervalo normal simples, porque com n pequeno e proporcao perto da
    borda o intervalo normal escapa de [0, 1] e deixa de ser confiavel. Aqui isso importa: subgrupo com 24
    positivos e' onde a diferenca de recall parece dramatica e pode ser sorteio.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = acertos / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meio = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centro - meio), min(1.0, centro + meio))


def fatiar(df: pd.DataFrame, y, p, limiar: float, chave: pd.Series, nome: str) -> None:
    base = metricas.matriz(y, p, limiar)
    lo_g, hi_g = wilson(base["vp"], base["vp"] + base["fn"])
    print(f"\n{nome}")
    print(f"   {'grupo':<24}{'n':>7}{'pos':>6}{'recall':>9}{'IC 95% do recall':>20}"
          f"{'precisao':>10}{'negados':>9}{'alerta':>9}")
    for g in sorted(chave.unique(), key=str):
        m = chave == g
        if m.sum() == 0:
            continue
        r = metricas.matriz(y[m], p[m], limiar)
        pos = r["vp"] + r["fn"]
        lo, hi = wilson(r["vp"], pos)
        negados = r["vp"] + r["fp"]
        # Alerta so' quando o intervalo do grupo NAO alcanca o recall geral: ai' a
        # diferenca nao se explica por tamanho de amostra. Grupo largo demais no
        # intervalo vira "incerto", que e' informacao diferente de "pior".
        if hi < base["recall"]:
            alerta = "PIOR"
        elif hi - lo > 0.30:
            alerta = "incerto"
        else:
            alerta = ""
        print(f"   {str(g):<24}{int(m.sum()):>7,}{pos:>6}{r['recall']:>9.3f}"
              f"{f'[{lo:.3f} , {hi:.3f}]':>20}{r['precisao']:>10.3f}{negados:>9,}{alerta:>9}")
    print(f"   {'(geral)':<24}{len(y):>7,}{base['vp'] + base['fn']:>6}"
          f"{base['recall']:>9.3f}{f'[{lo_g:.3f} , {hi_g:.3f}]':>20}"
          f"{base['precisao']:>10.3f}{base['vp'] + base['fp']:>9,}")


def main() -> None:
    partes = prepare.conjuntos()
    X_tr, y_tr = prepare.X_y(partes["treino"])
    X_cal, y_cal = prepare.X_y(partes["calibracao"])
    teste = partes["teste"]
    X_te, y_te = prepare.X_y(teste)

    modelo = logistica([c for c in X_tr.columns if c not in CATEGORICAS]).fit(X_tr, y_tr)
    p_cal = modelo.predict_proba(X_cal)[:, 1]
    p_te = modelo.predict_proba(X_te)[:, 1]
    t_star, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(RAZAO))
    piso = metricas.pisos(y_te)

    titulo(f"A MATRIZ, NO LIMIAR OPERACIONAL t = {t_star:.4f}  (1 FN = {RAZAO} FPs)")
    for nome, t in [("limiar padrao 0,5", 0.5), (f"limiar operacional {t_star:.4f}", t_star)]:
        m = metricas.matriz(y_te, p_te, t)
        print(f"\n   {nome}")
        print(f"      {'':<22}{'previsto INADIMPLENTE':>24}{'previsto PAGADOR':>20}")
        print(f"      {'real INADIMPLENTE':<22}{'VP = ' + str(m['vp']):>24}{'FN = ' + str(m['fn']):>20}")
        print(f"      {'real PAGADOR':<22}{'FP = ' + str(m['fp']):>24}{'VN = ' + str(m['vn']):>20}")
        print(f"      recall {m['recall']:.3f} | precisao {m['precisao']:.3f} | "
              f"negados {m['vp'] + m['fp']:,} de {len(y_te):,}")

    titulo("AS METRICAS, CADA UMA COM O SEU PISO")
    pred = (p_te >= t_star).astype(int)
    m = metricas.matriz(y_te, p_te, t_star)
    espec = m["vn"] / (m["vn"] + m["fp"])
    linhas = [
        ("Precisao (dos negados, quantos dariam calote)", m["precisao"], piso["prevalencia"],
         "quem nega ao acaso acerta a prevalencia"),
        ("Recall (dos caloteiros, quantos retidos)", m["recall"], 0.0, "o modelo burro nao acha nenhum"),
        ("Especificidade (dos bons, quantos em paz)", espec, 1.0, "o modelo burro deixa TODOS em paz"),
        ("Balanced accuracy", balanced_accuracy_score(y_te, pred), 0.5, "acaso"),
        ("MCC", matthews_corrcoef(y_te, pred), 0.0, "acaso"),
        ("AUPRC", metricas.linha("", y_te, p_te)["auprc"], piso["auprc"], "= prevalencia"),
        ("AUC-ROC", metricas.linha("", y_te, p_te)["auc"], 0.5, "acaso"),
    ]
    print(f"   {'metrica':<46}{'valor':>9}{'piso':>9}   quem alcanca o piso")
    for nome, valor, p_, quem in linhas:
        print(f"   {nome:<46}{valor:>9.3f}{p_:>9.3f}   {quem}")
    print("\n   A acurácia nao esta' nesta tabela de proposito: no limiar operacional ela")
    print(f"   cai para {(m['vp'] + m['vn']) / len(y_te):.4f}, abaixo do piso de {piso['acuracia']:.4f} — e isso e'")
    print("   esperado, nao defeito: a troca e' de acuracia por recall de caso raro.")

    titulo("FATIAMENTO — a media esconde falha localizada")
    fatiar(teste, y_te, p_te, t_star, teste["regiao"], "por regiao")
    fatiar(teste, y_te, p_te, t_star, teste["canal"], "por canal")
    fatiar(teste, y_te, p_te, t_star,
           pd.cut(teste["idade"], [17, 25, 35, 50, 100],
                  labels=["18-25", "26-35", "36-50", "51+"]).astype(str), "por faixa de idade")
    fatiar(teste, y_te, p_te, t_star,
           pd.qcut(teste["valor_solicitado"], 4,
                   labels=["valor Q1", "valor Q2", "valor Q3", "valor Q4"]).astype(str),
           "por faixa de valor solicitado")
    fatiar(teste, y_te, p_te, t_star,
           teste["possui_imovel"].map({0: "sem imovel", 1: "com imovel"}), "por garantia")

    titulo("O ACHADO DO PIOR SUBGRUPO SOBREVIVE A' ESCOLHA DO LIMIAR?")
    print("Um achado que so' existe num limiar e' artefato da escolha, nao propriedade do")
    print("modelo. Entao: o mesmo teste, repetido em cada cenario de custo da faixa.\n")
    pior = min(sorted(teste["regiao"].unique()),
               key=lambda g: metricas.matriz(y_te[(teste["regiao"] == g).to_numpy()],
                                             p_te[(teste["regiao"] == g).to_numpy()],
                                             t_star)["recall"])
    sel = (teste["regiao"] == pior).to_numpy()
    print(f"   pior subgrupo no cenario de referencia: {pior}\n")
    print(f"   {'razao':>7}{'limiar':>9}{'recall geral':>15}{f'recall {pior}':>16}"
          f"{'IC 95%':>18}   diferenca")
    reais, inconclusivas = 0, []
    for razao in metricas.FAIXA_RAZOES:
        t_r, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(razao))
        g = metricas.matriz(y_te, p_te, t_r)
        s = metricas.matriz(y_te[sel], p_te[sel], t_r)
        lo, hi = wilson(s["vp"], s["vp"] + s["fn"])
        real = hi < g["recall"]
        reais += real
        if not real:
            inconclusivas.append(f"{razao}:1 (recall geral {g['recall']:.2f})")
        print(f"   {razao:>6}x{t_r:>9.3f}{g['recall']:>15.3f}{s['recall']:>16.3f}"
              f"{f'[{lo:.2f} , {hi:.2f}]':>18}   {'REAL' if real else 'inconclusiva'}")
    print(f"\n   Real em {reais} dos {len(metricas.FAIXA_RAZOES)} cenarios.")
    if inconclusivas:
        print(f"   Inconclusiva em: {', '.join(inconclusivas)}.")
        print("   Leia o recall geral ao lado: onde ele e' muito baixo, o modelo nao acha")
        print("   quase ninguem em lugar nenhum, e nao sobra diferenca para medir.")

    titulo("COMO LER ESTE FATIAMENTO SEM EXAGERAR")
    print("   Grupo com menos de 30 positivos nao sustenta recall: um unico caso muda a")
    print("   terceira casa. Por isso a coluna de alerta separa 'incerto' (intervalo largo)")
    print("   de 'PIOR' (intervalo inteiro abaixo do recall geral).")
    print("   E recall baixo num grupo nao prova discriminacao: prova que o modelo tem")
    print("   menos evidencia ali. A pergunta seguinte e' se o grupo recebe decisao pior")
    print("   por isso — e essa e' pergunta de politica, nao de metrica.")


if __name__ == "__main__":
    main()
