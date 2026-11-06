"""Que recorte merece entrar na seção 7 do relatório? Todos, medidos.

Nomear UM subgrupo relevante por intuicao e' onde mora o viés de quem escolhe: olha o resultado, acha o recorte que dá manchete, e apresenta como
se tivesse escolhido antes. Entao aqui se mede TODOS os recortes que o arquivo permite,
nas duas perguntas que importam — e que sao diferentes:

  1. O modelo PROTEGE menos esse grupo?  -> recall abaixo do geral
     Quem paga: a pessoa do grupo cujo calote passa, e o banco.

  2. O modelo BARRA mais esse grupo?     -> taxa de negacao acima da geral
     Quem paga: a pessoa do grupo que teria pago e levou nao.

Um grupo pode ser pior numa e igual na outra. Sao perguntas distintas, e tratar as duas
como "desempenho por subgrupo" e' o jeito mais comum de esconder a segunda.

CRITERIO: so' conta como diferenca quando o intervalo de 95% do grupo NAO alcanca o valor
geral. Sem isso, subgrupo pequeno sempre parece extremo, e a lista vira ruido ordenado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import metricas
import prepare
from camada_1_erro_e_subgrupos import wilson
from modelo_e_baselines import logistica
from prepare import CATEGORICAS


def titulo(t: str) -> None:
    print(f"\n{'=' * 86}\n{t}\n{'=' * 86}")


def recortes(teste: pd.DataFrame) -> dict[str, pd.Series]:
    """Todo recorte que o arquivo permite, com o motivo de cada um estar na lista."""
    return {
        "regiao": teste["regiao"],
        "canal": teste["canal"],
        "faixa de idade": pd.cut(teste["idade"], [17, 25, 35, 50, 100],
                                 labels=["18-25", "26-35", "36-50", "51+"]).astype(str),
        "faixa de renda": pd.qcut(teste["renda_mensal"], 4,
                                  labels=["renda Q1", "renda Q2", "renda Q3", "renda Q4"]).astype(str),
        "faixa de valor": pd.qcut(teste["valor_solicitado"], 4,
                                  labels=["valor Q1", "valor Q2", "valor Q3", "valor Q4"]).astype(str),
        "faixa de score": pd.qcut(teste["score_bureau"], 4,
                                  labels=["score Q1", "score Q2", "score Q3", "score Q4"]).astype(str),
        "garantia": teste["possui_imovel"].map({0: "sem imovel", 1: "com imovel"}),
        "prazo": teste["num_parcelas"].astype(str) + "x",
        "atrasos em 12m": teste["historico_atraso_12m"].astype(str) + " atraso(s)",
        "relacao com o banco": pd.cut(teste["tempo_relacionamento_banco_meses"],
                                      [-1, 6, 24, 60, 10**4],
                                      labels=["ate 6m", "6-24m", "2-5a", "5a+"]).astype(str),
    }


def main() -> None:
    partes = prepare.conjuntos()
    X_tr, y_tr = prepare.X_y(partes["treino"])
    X_cal, y_cal = prepare.X_y(partes["calibracao"])
    teste = partes["teste"]
    X_te, y_te = prepare.X_y(teste)
    y_te = np.asarray(y_te)

    modelo = logistica([c for c in X_tr.columns if c not in CATEGORICAS]).fit(X_tr, y_tr)
    p_cal = modelo.predict_proba(X_cal)[:, 1]
    p_te = modelo.predict_proba(X_te)[:, 1]
    t, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(metricas.RAZAO_REFERENCIA))
    geral = metricas.matriz(y_te, p_te, t)
    taxa_geral = (geral["vp"] + geral["fp"]) / len(y_te)

    print(f"limiar operacional t = {t:.4f} (razao {metricas.RAZAO_REFERENCIA}:1)")
    print(f"recall geral {geral['recall']:.3f} | taxa de negacao geral {taxa_geral:.1%} "
          f"| {len(y_te):,} solicitacoes, {int(y_te.sum())} inadimplentes")

    achados = []
    for nome, chave in recortes(teste).items():
        titulo(f"RECORTE: {nome}")
        print(f"   {'grupo':<18}{'n':>7}{'pos':>6}{'recall':>9}{'IC do recall':>18}"
              f"{'negados':>10}{'IC da negacao':>18}   alerta")
        for g in sorted(chave.unique(), key=str):
            sel = (chave == g).to_numpy()
            if sel.sum() < 30:
                continue
            r = metricas.matriz(y_te[sel], p_te[sel], t)
            pos = r["vp"] + r["fn"]
            lo_r, hi_r = wilson(r["vp"], pos) if pos else (float("nan"), float("nan"))
            negados = r["vp"] + r["fp"]
            lo_n, hi_n = wilson(negados, int(sel.sum()))
            # Duas perguntas, dois alertas. "PROTEGE MENOS": o intervalo do recall do
            # grupo inteiro fica abaixo do recall geral. "BARRA MAIS": o intervalo da
            # taxa de negacao fica inteiro acima da taxa geral.
            marcas = []
            if pos and hi_r < geral["recall"]:
                marcas.append("PROTEGE MENOS")
            if lo_n > taxa_geral:
                marcas.append("BARRA MAIS")
            if marcas:
                achados.append({"recorte": nome, "grupo": str(g), "n": int(sel.sum()),
                                "positivos": pos, "recall": r["recall"],
                                "negados": negados / sel.sum(), "alerta": " + ".join(marcas)})
            print(f"   {str(g):<18}{int(sel.sum()):>7,}{pos:>6}{r['recall']:>9.3f}"
                  f"{f'[{lo_r:.2f} , {hi_r:.2f}]':>18}{negados / sel.sum():>10.1%}"
                  f"{f'[{lo_n:.2f} , {hi_n:.2f}]':>18}   {' + '.join(marcas)}")

    titulo("O QUE SOBREVIVEU AO INTERVALO DE CONFIANCA")
    if not achados:
        print("   nenhum recorte com diferenca sustentada.")
    else:
        tab = pd.DataFrame(achados)
        print(tab.to_string(index=False,
                            formatters={"recall": "{:.3f}".format, "negados": "{:.1%}".format}))
    print()
    print("   Como levar isto para a secao 7: o recorte que entra no relatorio e' o que tem")
    print("   diferenca sustentada E consequencia para alguem. Um grupo que e' barrado mais")
    print("   porque tem risco maior de verdade nao e' o mesmo caso de um grupo que e'")
    print("   barrado mais E protegido menos — o segundo recebe o pior dos dois mundos.")


if __name__ == "__main__":
    main()
