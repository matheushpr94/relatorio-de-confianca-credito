"""Quando o modelo diz 0,30, é 0,30 mesmo?

Camada 3. Um modelo e' calibrado se, entre os casos em que ele disse 0,30, cerca de 30%
forem mesmo inadimplentes. Ordenar bem e estimar bem sao propriedades independentes: a
regra do score do bureau ordena bem acima do acaso e tem Brier PIOR que o modelo burro
(ver outputs/03).

Calibra na CALIBRACAO e reporta no TESTE. Calibrar e reportar no mesmo conjunto e'
vazamento, e e' a regra que nao tem excecao.

LIMITACAO ASSUMIDA: o mesmo conjunto de calibracao serve para tres decisoes — escolher o
modelo, ajustar o calibrador e escolher o limiar. O certo seria um quarto conjunto. O
numero relatado no teste continua honesto, porque o teste nao participou de nenhuma das
tres; o risco e' o limiar estar levemente sobreajustado a' calibracao.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

import metricas
import prepare
from modelo_e_baselines import logistica
from prepare import CATEGORICAS

RAZAO = metricas.RAZAO_REFERENCIA


def titulo(t: str) -> None:
    print(f"\n{'=' * 84}\n{t}\n{'=' * 84}")


def confiabilidade(y, p, n_bins: int = 8) -> None:
    """Diagrama de confiabilidade em texto: bins de mesma populacao."""
    y, p = np.asarray(y), np.asarray(p)
    bordas = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, bordas[1:-1]), 0, len(bordas) - 2)
    print(f"   {'faixa de probabilidade':<28}{'n':>7}{'previsto':>11}{'observado':>12}"
          f"{'diferenca':>12}  leitura")
    for b in range(len(bordas) - 1):
        m = idx == b
        if not m.sum():
            continue
        prev, obs = p[m].mean(), y[m].mean()
        d = obs - prev
        leitura = "sub-confiante" if d > 0.01 else ("super-confiante" if d < -0.01 else "ok")
        print(f"   [{bordas[b]:.3f} , {bordas[b + 1]:.3f}]{'':<10}{int(m.sum()):>7,}"
              f"{prev:>11.3f}{obs:>12.3f}{d:>+12.3f}  {leitura}")


def versoes_calibradas(X_tr, y_tr, X_cal, y_cal, X_te) -> dict[str, tuple]:
    """Para cada versao do modelo, o par (escore na calibracao, escore no teste).

    Fica como funcao para o relatorio usar os MESMOS numeros — dois lugares calculando a
    mesma coisa e' como um numero de prosa envelhece sem ninguem ver.
    """
    modelo = logistica([c for c in X_tr.columns if c not in CATEGORICAS]).fit(X_tr, y_tr)
    iso = CalibratedClassifierCV(FrozenEstimator(modelo), method="isotonic").fit(X_cal, y_cal)
    platt = CalibratedClassifierCV(FrozenEstimator(modelo), method="sigmoid").fit(X_cal, y_cal)
    return {
        "sem calibrar": (modelo.predict_proba(X_cal)[:, 1], modelo.predict_proba(X_te)[:, 1]),
        "Platt": (platt.predict_proba(X_cal)[:, 1], platt.predict_proba(X_te)[:, 1]),
        "Isotonic": (iso.predict_proba(X_cal)[:, 1], iso.predict_proba(X_te)[:, 1]),
    }


def main() -> None:
    partes = prepare.conjuntos()
    X_tr, y_tr = prepare.X_y(partes["treino"])
    X_cal, y_cal = prepare.X_y(partes["calibracao"])
    X_te, y_te = prepare.X_y(partes["teste"])

    versoes = versoes_calibradas(X_tr, y_tr, X_cal, y_cal, X_te)
    piso = metricas.pisos(y_te)

    titulo("DIAGRAMA DE CONFIABILIDADE — o modelo sem calibrar, no teste")
    confiabilidade(y_te, versoes["sem calibrar"][1])
    print(f"\n   Acima da diagonal (observado > previsto) = sub-confiante: o modelo promete")
    print("   menos risco do que existe. Abaixo = super-confiante.")

    titulo("AS TRES VERSOES, MEDIDAS NO TESTE")
    print(f"   {'versao':<16}{'ECE':>9}{'ECE quantil':>14}{'Brier':>10}{'AUC':>9}"
          f"{'Brier/piso':>12}")
    for nome, (_, p_te) in versoes.items():
        L = metricas.linha(nome, y_te, p_te)
        print(f"   {nome:<16}{L['ece']:>9.4f}{L['ece_q']:>14.4f}{L['brier']:>10.5f}"
              f"{L['auc']:>9.4f}{L['brier'] / piso['brier']:>12.3f}")
    print(f"\n   piso: ECE 0 (alcancado pelo modelo que devolve {piso['prevalencia']:.2%} para")
    print(f"   todo mundo), Brier {piso['brier']:.4f}, AUC 0,5.")
    print("   ECE sozinho nao decide nada: o inutil tem ECE zero. Ele so' vale ao lado da AUC.")

    titulo("A CALIBRACAO MUDOU A ORDENACAO?")
    base = metricas.linha("", y_te, versoes["sem calibrar"][1])["auc"]
    for nome, (_, p_te) in versoes.items():
        d = metricas.linha("", y_te, p_te)["auc"] - base
        print(f"   {nome:<16}AUC {metricas.linha('', y_te, p_te)['auc']:.4f}  ({d:+.4f})")
    print("\n   Platt e' monotona, entao NAO pode mudar a ordem — a AUC tem que ficar igual.")
    print("   Isotonic e' monotona por partes e empata faixas inteiras: ela pode mexer na")
    print("   terceira casa, e o que muda e' empate, nao ordem.")

    titulo("DEPOIS DE CALIBRAR, O LIMIAR AINDA VALE?")
    print("   Nao: calibrar desloca a escala. O limiar tem que ser recalculado, senao a")
    print(f"   operacao muda sem ninguem decidir. Com 1 FN = {RAZAO} FPs:\n")
    print(f"   {'versao':<16}{'limiar':>10}{'recall':>9}{'precisao':>10}{'negados':>10}")
    for nome, (p_cal, p_te) in versoes.items():
        t, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(RAZAO))
        m = metricas.matriz(y_te, p_te, t)
        print(f"   {nome:<16}{t:>10.4f}{m['recall']:>9.3f}{m['precisao']:>10.3f}"
              f"{m['vp'] + m['fp']:>10,}")
    print("\n   Se as tres linhas entregam operacao parecida, a calibracao nao mudou a")
    print("   DECISAO — mudou o numero apresentado a quem decide. Isso importa")
    print("   quando a probabilidade entra em conta: preco, provisao, fila de analise.")


if __name__ == "__main__":
    main()
