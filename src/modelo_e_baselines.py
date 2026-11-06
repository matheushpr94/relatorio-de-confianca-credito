"""O modelo caro ganha de quê? Cinco candidatos, comparados na calibração.

Ordem que o METODO exige: o burro primeiro, depois a heuristica que qualquer analista
faria, depois o modelo simples, e so' entao o caro. Sem isso, "AUC 0,80" nao significa
nada — pode ser pior que olhar uma coluna.

Os cinco:
  1. constante        — sempre "nao vai dar calote". E' o piso de acuracia, e recall zero.
  2. regra do score   — risco = quanto menor o score do bureau. Uma coluna, zero treino.
  3. logistica        — o padrao de credito: a negativa precisa ser explicavel.
  4. floresta         — Random Forest, o candidato "padrao" de muito tutorial.
  5. boosting         — HistGradientBoosting, ja' dentro do scikit-learn.

DECISAO: sem `class_weight`, sem SMOTE, sem reamostrar. Desbalanceamento se resolve no
LIMIAR, que e' a camada 2 — reponderar o treino desloca a probabilidade que a camada 3
vai medir, e ai' a calibracao passa a corrigir um problema criado pelo proprio treino.

O conjunto de TESTE nao e' tocado aqui. A escolha acontece na calibracao; o teste so' e'
aberto no relatorio final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import metricas
import prepare
from prepare import ALVO, CATEGORICAS


def titulo(t: str) -> None:
    print(f"\n{'=' * 84}\n{t}\n{'=' * 84}")


def logistica(numericas: list[str]) -> Pipeline:
    """One-hot nas categoricas, padronizacao nas numericas — a regressao mede distancia."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAS),
        ("num", StandardScaler(), numericas),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])


def boosting(numericas: list[str]) -> Pipeline:
    """Arvore nao mede distancia: categorica vira codigo, e o modelo sabe que e' categoria."""
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAS),
        ("num", "passthrough", numericas),
    ])
    modelo = HistGradientBoostingClassifier(
        categorical_features=list(range(len(CATEGORICAS))), random_state=prepare.SEMENTE)
    return Pipeline([("pre", pre), ("clf", modelo)])


def floresta(numericas: list[str]) -> Pipeline:
    """Random Forest, sem ajuste de profundidade.

    Entra na comparacao para nao ser descartado sem medir. Com folhas puras, a floresta
    tende a DECORAR o treino; a secao de sobreajuste abaixo mede isso diretamente.
    """
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("num", "passthrough", numericas),
    ])
    modelo = RandomForestClassifier(n_estimators=200, random_state=prepare.SEMENTE, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", modelo)])


def main() -> None:
    partes = prepare.conjuntos()
    X_tr, y_tr = prepare.X_y(partes["treino"])
    X_cal, y_cal = prepare.X_y(partes["calibracao"])
    numericas = [c for c in X_tr.columns if c not in CATEGORICAS]

    print(f"treino {len(y_tr):,} | calibracao {len(y_cal):,} | teste {len(partes['teste']):,} "
          f"(fechado)")
    piso = metricas.pisos(y_cal)
    print(f"prevalencia na calibracao: {piso['prevalencia']:.2%}")

    titulo("OS CINCO CANDIDATOS, MEDIDOS NA CALIBRACAO")
    escores = {}
    escores["constante"] = np.full(len(y_cal), y_tr.mean())
    # A regra: risco cresce quando o score do bureau cai. Normalizada para virar um numero
    # entre 0 e 1 — e' ordenacao, nao probabilidade, e a camada 3 vai cobrar isso.
    escores["regra do score"] = (850 - X_cal["score_bureau"]) / (850 - 300)
    for nome, modelo in [("logistica", logistica(numericas)), ("floresta", floresta(numericas)),
                         ("boosting", boosting(numericas))]:
        modelo.fit(X_tr, y_tr)
        escores[nome] = modelo.predict_proba(X_cal)[:, 1]

    tabela = pd.DataFrame([metricas.linha(n, y_cal, p) for n, p in escores.items()])
    tabela["auprc/piso"] = tabela["auprc"] / piso["auprc"]
    tabela["brier/piso"] = tabela["brier"] / piso["brier"]
    print()
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\npiso: AUC {piso['auc']:.3f} | AUPRC {piso['auprc']:.4f} (= prevalencia) | "
          f"Brier {piso['brier']:.4f} | ECE 0")
    print("\nComo ler: AUC e AUPRC medem ORDENACAO; Brier e ECE medem se o numero devolvido")
    print("vale como probabilidade. Um modelo pode ordenar bem e mentir na probabilidade.")

    titulo("O QUE CADA UM CUSTA EM ACURACIA — a metrica que engana")
    for nome, p in escores.items():
        acuracia = ((p >= 0.5).astype(int) == y_cal).mean()
        pegou = int(((p >= 0.5) & (y_cal == 1)).sum())
        print(f"   {nome:<16} acuracia no limiar 0,5: {acuracia:.4f}   "
              f"inadimplentes encontrados: {pegou:>4} de {int(y_cal.sum())}")
    print(f"\n   O piso de acuracia e' {piso['acuracia']:.4f}. Quem fica colado nele e nao acha")
    print("   ninguem e' o modelo que sempre responde 'paga': acuracia alta, utilidade zero.")

    titulo("O BOOSTING GANHOU DA LOGISTICA? E DA REGRA DE UMA COLUNA?")
    base = tabela.set_index("modelo")
    for a, b in [("boosting", "logistica"), ("floresta", "logistica"),
                 ("logistica", "regra do score"), ("boosting", "regra do score")]:
        d_auprc = base.loc[a, "auprc"] - base.loc[b, "auprc"]
        print(f"   {a:<16} - {b:<16} AUPRC {d_auprc:+.4f}   "
              f"AUC {base.loc[a, 'auc'] - base.loc[b, 'auc']:+.4f}")
    print("\n   Se o modelo complexo nao ganha da logistica por margem clara, a logistica")
    print("   fica: em credito, a negativa precisa ser explicada.")

    titulo("SOBREAJUSTE: treino contra calibracao")
    for nome, modelo in [("logistica", logistica(numericas)), ("floresta", floresta(numericas)),
                         ("boosting", boosting(numericas))]:
        modelo.fit(X_tr, y_tr)
        p_tr = modelo.predict_proba(X_tr)[:, 1]
        p_cal = modelo.predict_proba(X_cal)[:, 1]
        print(f"   {nome:<12} AUPRC treino {metricas.linha(nome, y_tr, p_tr)['auprc']:.4f}  "
              f"calibracao {metricas.linha(nome, y_cal, p_cal)['auprc']:.4f}")
    print("\n   Distancia grande entre as duas colunas = o modelo decorou o treino.")


if __name__ == "__main__":
    main()
