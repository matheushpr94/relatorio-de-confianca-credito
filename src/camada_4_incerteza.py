"""O modelo sabe quando não sabe? E quem atende quando ele diz "não sei"?

Camada 4. Abster e' devolver o caso dificil para uma pessoa. So' vale se existir pessoa.

A ARMADILHA: confianca nao e' `max(p, 1-p)`. Isso mede distancia do 0,5, e o limiar
operacional aqui fica abaixo de 0,1. Um caso com probabilidade 0,05 esta' PERTO da
fronteira, nao longe dela. A margem certa e' |p - t*|.

Segunda armadilha: `np.quantile` sobre a margem falha quando ha' muitos escores
empatados. A selecao sai por `argsort`.
"""
from __future__ import annotations

import numpy as np

import metricas
import prepare
from modelo_e_baselines import logistica
from prepare import CATEGORICAS

RAZAO = metricas.RAZAO_REFERENCIA
COBERTURAS = [1.0, 0.98, 0.95, 0.90, 0.80]


def titulo(t: str) -> None:
    print(f"\n{'=' * 84}\n{t}\n{'=' * 84}")


def main() -> None:
    partes = prepare.conjuntos()
    X_tr, y_tr = prepare.X_y(partes["treino"])
    X_cal, y_cal = prepare.X_y(partes["calibracao"])
    X_te, y_te = prepare.X_y(partes["teste"])

    modelo = logistica([c for c in X_tr.columns if c not in CATEGORICAS]).fit(X_tr, y_tr)
    p_cal = modelo.predict_proba(X_cal)[:, 1]
    p_te = modelo.predict_proba(X_te)[:, 1]
    t_star, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(RAZAO))
    y_te = np.asarray(y_te)
    n = len(y_te)

    titulo(f"A MARGEM CERTA E' ATE' O LIMIAR OPERACIONAL ({t_star:.4f}), NAO ATE' 0,5")
    margem_certa = np.abs(p_te - t_star)
    margem_errada = np.abs(p_te - 0.5)
    k10 = int(round(n * 0.10))
    concordam = len(set(np.argsort(margem_certa)[:k10]) & set(np.argsort(margem_errada)[:k10])) / k10
    print(f"   Pela distancia ao 0,5, os 10% 'mais incertos' seriam outras pessoas:")
    print(f"   as duas listas tem {concordam:.1%} das pessoas em comum.")
    piores_certo = np.argsort(margem_certa)[:k10]
    piores_errado = np.argsort(margem_errada)[:k10]
    print(f"   probabilidade mediana dos 10% mais incertos pela margem CERTA:  "
          f"{np.median(p_te[piores_certo]):.4f}")
    print(f"   pela margem ERRADA (distancia ao 0,5):                          "
          f"{np.median(p_te[piores_errado]):.4f}")
    print(f"   positivos reais capturados na fila da margem certa:  "
          f"{int(y_te[piores_certo].sum())} de {int(y_te.sum())}")
    print(f"   na fila da margem errada:                            "
          f"{int(y_te[piores_errado].sum())} de {int(y_te.sum())}")

    titulo("CURVA RISCO-COBERTURA — o que se ganha ao deixar de responder")
    ordem = np.argsort(margem_certa)
    pred = (p_te >= t_star).astype(int)
    print(f"   {'cobertura':>10}{'respondidos':>13}{'revisao humana':>16}{'precisao':>10}"
          f"{'recall':>9}{'calotes na fila':>18}")
    for cob in COBERTURAS:
        k = int(round(n * (1 - cob)))
        fica = np.ones(n, bool)
        fica[ordem[:k]] = False
        m = metricas.matriz(y_te[fica], p_te[fica], t_star)
        print(f"   {cob:>9.0%}{int(fica.sum()):>13,}{k:>16,}{m['precisao']:>10.3f}"
              f"{m['recall']:>9.3f}{int(y_te[~fica].sum()):>12} de {int(y_te.sum())}")

    titulo("O QUE ISSO CUSTA EM GENTE")
    print("   O arquivo nao tem data, entao nao existe 'casos por dia' honesto aqui: a")
    print("   conta depende do volume de solicitacoes, que e' informacao de operacao.")
    print("   Entao a conta vai por mil solicitacoes, e quem opera multiplica.\n")
    for cob in [0.98, 0.95, 0.90, 0.80]:
        print(f"   cobertura {cob:.0%}  ->  {int(round((1 - cob) * 1000)):>4} analises humanas "
              f"por 1.000 solicitacoes")
    print("\n   Com 18.000 solicitacoes, 90% de cobertura seriam 1.800 analises. Se isso for")
    print("   um mes, sao ~60 por dia util; se for um dia, e' outra conversa inteira. A")
    print("   pergunta 'em que periodo?' vira lacuna do relatorio.")

    titulo("AS QUATRO PERGUNTAS QUE UMA POLITICA DE ABSTENCAO PRECISA RESPONDER")
    for i, (p, resposta) in enumerate([
        ("Quem revisa?", "NAO RESPONDIDA — sem pessoa designada, a politica fica no papel"),
        ("Quantos por dia?", "depende do volume; a tabela acima da' a conta por mil"),
        ("Em quanto tempo?", "NAO RESPONDIDA — analise que demora mais que a concessao nao serve"),
        ("O revisor acerta mais que o modelo?",
         "NAO RESPONDIDA — e se nao acertar, encaminhar PIORA o resultado"),
    ], 1):
        print(f"   {i}. {p:<38} {resposta}")
    print("\n   Tres das quatro sem resposta. Abster 10% sem ninguem para revisar deixa o")
    print("   sistema pior do que responder tudo — por isso a recomendacao do relatorio")
    print("   nao pode ser 'com revisao humana' antes de alguem assumir a fila.")


if __name__ == "__main__":
    main()
