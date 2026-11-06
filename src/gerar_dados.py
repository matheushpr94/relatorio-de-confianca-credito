"""Gera o dado SINTÉTICO de solicitações de crédito usado por todo o pipeline.

POR QUE SINTÉTICO. O exercício original usou um arquivo entregue em aula, sem licença
declarada. Ele não é meu e não entra neste repositório, nem nada derivado dele linha a
linha. Este gerador reproduz apenas o que é público por natureza: o SCHEMA (15 colunas,
com os domínios declarados em `prepare.ESQUEMA`) e a ordem de grandeza da prevalência.
Toda distribuição e todo coeficiente abaixo foram escritos à mão.

O QUE FOI PLANTADO, DE PROPÓSITO, E PRECISA SER DECLARADO:

1. `score_bureau` é a variável mais informativa. Assim a "régua de uma coluna" (ordenar
   só pelo score) é um baseline forte, como costuma ser em crédito.
2. Na região NORTE o risco depende pouco das variáveis observadas e muito de um fator
   LATENTE que não está no arquivo. A prevalência do Norte fica parecida com a das outras
   regiões; o que muda é que o modelo tem menos evidência para separar quem paga de quem
   não paga ali. Este é o efeito que a análise de subgrupos deve encontrar. Se não
   encontrasse, o pipeline estaria quebrado.

Nada aqui descreve um mercado real. Os números da análise valem como demonstração do
método, não como afirmação sobre crédito no Brasil.

    uv run python src/gerar_dados.py     -> data/solicitacoes_sinteticas.csv
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data" / "solicitacoes_sinteticas.csv"

# A semente e os dois pesos abaixo foram escolhidos entre 8 sementes x 2 configuracoes,
# olhando o resultado: o alvo era prevalencia perto de 7,7% e um efeito no Norte visivel mas
# nao caricato. Declarado porque escolher o gerador vendo a saida e' exatamente o tipo de
# grau de liberdade que precisa ficar a' vista. Isso NAO afeta a validade do metodo (o
# efeito foi plantado de qualquer jeito); afeta so' quais numeros aparecem no README.
SEMENTE = 5
N = 18_000
PREVALENCIA_ALVO = 0.077

REGIOES = {"Sudeste": 0.42, "Nordeste": 0.26, "Sul": 0.14, "Norte": 0.10, "Centro-Oeste": 0.08}
CANAIS = {"App": 0.54, "Agência": 0.30, "Correspondente": 0.16}
PARCELAS = {12: 0.14, 24: 0.24, 36: 0.31, 48: 0.21, 60: 0.10}

# O efeito plantado no Norte: quanto do sinal observado sobra, e o peso do fator latente.
NORTE_SINAL_OBSERVADO = 0.15
NORTE_PESO_LATENTE = 1.6
OUTRAS_PESO_LATENTE = 0.4


def gerar(semente: int = SEMENTE, n: int = N, sinal_norte: float | None = None,
          latente_norte: float | None = None, latente_outras: float | None = None) -> pd.DataFrame:
    sinal_norte = NORTE_SINAL_OBSERVADO if sinal_norte is None else sinal_norte
    latente_norte = NORTE_PESO_LATENTE if latente_norte is None else latente_norte
    latente_outras = OUTRAS_PESO_LATENTE if latente_outras is None else latente_outras
    rng = np.random.default_rng(semente)

    regiao = rng.choice(list(REGIOES), size=n, p=list(REGIOES.values()))
    canal = rng.choice(list(CANAIS), size=n, p=list(CANAIS.values()))
    idade = np.clip(np.round(rng.normal(38, 11.5, n)), 18, 75).astype(int)

    # Renda cresce um pouco com a idade; piso de R$ 1.200 (corte declarado do gerador).
    renda = np.exp(rng.normal(8.05 + 0.006 * (idade - 38), 0.52, n))
    renda = np.round(np.clip(renda, 1200, 30000), 2)

    # Tempos cabem na vida da pessoa: o schema checa isso, e o gerador respeita.
    teto_emprego = (idade - 14) * 12
    tempo_emprego = np.minimum(np.round(rng.exponential(40, n)), teto_emprego).astype(int)
    teto_banco = (idade - 18) * 12
    tempo_banco = np.minimum(np.round(rng.exponential(33, n)), teto_banco).astype(int)

    possui_imovel = (rng.random(n) < 0.18 + 0.006 * (idade - 18)).astype(int)
    atrasos = np.clip(rng.poisson(0.35, n), 0, 4)
    consultas = np.clip(rng.poisson(1.1, n), 0, 7)
    divida_renda = np.round(np.clip(rng.gamma(2.9, 0.112, n), 0.002, 1.12), 3)

    # Score do bureau carrega parte do histórico (atrasos, consultas) e ruído próprio.
    score = 655 - 38 * atrasos - 9 * consultas + 0.12 * tempo_banco + rng.normal(0, 85, n)
    score = np.clip(np.round(score), 300, 850).astype(int)

    valor = np.exp(rng.normal(9.75, 0.7, n)) * (renda / 3500) ** 0.4
    valor = np.round(np.clip(valor, 1500, 250000), 2)
    parcelas = rng.choice(list(PARCELAS), size=n, p=list(PARCELAS.values()))

    # Risco observável: o que um modelo consegue aprender a partir do arquivo.
    observado = (
        -0.95 * (score - 650) / 95
        + 0.30 * atrasos
        + 0.55 * (divida_renda - 0.32) / 0.19
        + 0.12 * consultas
        - 0.18 * np.log1p(tempo_emprego) / 2
        - 0.25 * possui_imovel
        - 0.20 * (np.log(renda) - 8.1) / 0.5
        + 0.10 * (parcelas - 36) / 14
    )
    latente = rng.normal(0, 1, n)
    norte = regiao == "Norte"
    risco = np.where(norte,
                     sinal_norte * observado + latente_norte * latente,
                     observado + latente_outras * latente)

    # Intercepto por bisseção para a prevalência alvo, separado por grupo (Norte e resto),
    # para o efeito plantado ser de SINAL e não de taxa: as duas taxas ficam parecidas.
    alvo = np.zeros(n, dtype=int)
    u = rng.random(n)
    for grupo in (norte, ~norte):
        r = risco[grupo]
        lo, hi = -10.0, 5.0
        for _ in range(60):
            meio = (lo + hi) / 2
            if (1 / (1 + np.exp(-(meio + r)))).mean() > PREVALENCIA_ALVO:
                hi = meio
            else:
                lo = meio
        p = 1 / (1 + np.exp(-(lo + r)))
        alvo[grupo] = (u[grupo] < p).astype(int)

    return pd.DataFrame({
        "id_solicitacao": np.arange(1, n + 1),
        "idade": idade,
        "renda_mensal": renda.astype(float),
        "tempo_emprego_meses": tempo_emprego,
        "score_bureau": score,
        "divida_renda": divida_renda.astype(float),
        "historico_atraso_12m": atrasos.astype(int),
        "consultas_bureau_6m": consultas.astype(int),
        "valor_solicitado": valor.astype(float),
        "num_parcelas": parcelas.astype(int),
        "possui_imovel": possui_imovel,
        "tempo_relacionamento_banco_meses": tempo_banco,
        "canal": canal,
        "regiao": regiao,
        "inadimplente": alvo,
    })


def main() -> None:
    df = gerar()
    DESTINO.parent.mkdir(exist_ok=True)
    df.to_csv(DESTINO, index=False, lineterminator="\n")
    sha = hashlib.sha256(DESTINO.read_bytes()).hexdigest()

    print("DADO SINTETICO — gerado por src/gerar_dados.py, nao e' dado real")
    print(f"semente {SEMENTE} | {len(df):,} linhas x {df.shape[1]} colunas")
    print(f"sha256 {sha}")
    print(f"prevalencia geral: {df['inadimplente'].mean():.2%}")
    print("\nEfeito PLANTADO na regiao Norte: mesma taxa de calote, sinal observavel")
    print(f"multiplicado por {NORTE_SINAL_OBSERVADO} e fator latente com peso "
          f"{NORTE_PESO_LATENTE} (outras regioes: {OUTRAS_PESO_LATENTE}).\n")
    tab = df.groupby("regiao")["inadimplente"].agg(linhas="size", positivos="sum", taxa="mean")
    print(tab.to_string(formatters={"taxa": "{:.2%}".format}))
    print("\nscore_bureau medio por desfecho (o sinal da regua de uma coluna):")
    print(df.groupby("inadimplente")["score_bureau"].mean().round(1).to_string())


if __name__ == "__main__":
    main()
