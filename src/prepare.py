"""Preparo canonico da tabela. TODA analise entra por aqui.

Nao existe "a tabela limpa": existe a tabela certa para cada pergunta. Este modulo expoe
as visoes como funcoes nomeadas, cada uma com a decisao documentada na docstring: QUANDO
usar, e por que nao usar as outras.

Regra: nenhum script de analise chama `read_csv`, `train_test_split` ou `drop_duplicates`
por conta propria. Dois scripts com sorteios diferentes produzem numeros que nao conversam,
e a comparacao entre eles vira ruido com cara de resultado.

O arquivo lido e' SINTETICO, gerado por `gerar_dados.py`. Ver o README.
"""
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

RAW = Path(__file__).resolve().parents[1] / "data/solicitacoes_sinteticas.csv"


# ---------------------------------------------------------------------------
# 1. O dominio de cada coluna, declarado UMA vez.
#
# Prosa num README nao falha nunca; schema falha em tempo de execucao. Por isso o
# dominio vive aqui, e o gerador sintetico e' obrigado a respeita-lo.
# ---------------------------------------------------------------------------
ALVO = "inadimplente"
CATEGORICAS = ["canal", "regiao"]
NAO_FEATURES = ["id_solicitacao", ALVO]

CANAIS = ["App", "Agência", "Correspondente"]
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

# O dominio e' o DEFENSAVEL, nao o observado. Declarar o observado faria o esquema
# aprovar tudo por construcao e nunca pegar nada.
ESQUEMA = DataFrameSchema(
    {
        "id_solicitacao": Column(int, Check.gt(0), unique=True),
        "idade": Column(int, Check.in_range(18, 100)),          # 18 e' o minimo legal para credito
        "renda_mensal": Column(float, Check.gt(0)),
        "tempo_emprego_meses": Column(int, Check.in_range(0, 720)),
        "score_bureau": Column(int, Check.in_range(300, 850)),  # escala 300-850
        "divida_renda": Column(float, Check.in_range(0, 10)),   # passa de 1: divida maior que a renda
        "historico_atraso_12m": Column(int, Check.in_range(0, 12)),
        "consultas_bureau_6m": Column(int, Check.in_range(0, 50)),
        "valor_solicitado": Column(float, Check.gt(0)),
        "num_parcelas": Column(int, Check.isin([12, 24, 36, 48, 60])),
        "possui_imovel": Column(int, Check.isin([0, 1])),
        "tempo_relacionamento_banco_meses": Column(int, Check.in_range(0, 1200)),
        "canal": Column(str, Check.isin(CANAIS)),
        "regiao": Column(str, Check.isin(REGIOES)),
        ALVO: Column(int, Check.isin([0, 1])),
    },
    checks=[
        # Coerencia entre colunas. Ninguem trabalha desde antes dos 14, nem tem conta em
        # banco desde antes dos 18. Violacao aqui e' sinal de colunas sorteadas
        # independentes uma da outra.
        Check(lambda df: df["tempo_emprego_meses"] <= (df["idade"] - 14) * 12,
              name="tempo_emprego cabe na vida do solicitante"),
        Check(lambda df: df["tempo_relacionamento_banco_meses"] <= (df["idade"] - 18) * 12,
              name="tempo_relacionamento cabe na vida adulta do solicitante"),
    ],
    strict=True,
    name="solicitacoes de credito (dominio defensavel)",
)


def fora_do_dominio(df: pd.DataFrame) -> pd.Index:
    """Indices das linhas que violam `ESQUEMA`. Vazio se a tabela estiver limpa.

    Levanta se o problema for de ESTRUTURA (coluna faltando, dtype errado) em vez de
    valor: isso nao se resolve descartando linha, e passar batido seria pior que falhar.
    """
    try:
        ESQUEMA.validate(df, lazy=True)
        return pd.Index([])
    except pa.errors.SchemaErrors as erro:
        casos = erro.failure_cases
        estruturais = casos[casos["index"].isna()]
        if not estruturais.empty:
            quais = estruturais[["column", "check"]].drop_duplicates().to_string(index=False)
            raise ValueError(f"tabela fora do esquema, nao e' problema de linha:\n{quais}") from erro
        return pd.Index(casos["index"].dropna().unique())


def incoerentes(df: pd.DataFrame) -> pd.Index:
    """Linhas em que as colunas de tempo se contradizem com a idade.

    No exercicio original, ~9% do arquivo caia aqui, concentrado em jovens; a decisao foi
    MANTER essas linhas e declarar a limitacao, porque descarta-las enviesaria o modelo
    contra o grupo que ele depois atenderia. O gerador sintetico respeita a coerencia,
    entao aqui o conjunto e' vazio, e o selftest confere isso.
    """
    fora_emprego = df["tempo_emprego_meses"] > (df["idade"] - 14) * 12
    fora_banco = df["tempo_relacionamento_banco_meses"] > (df["idade"] - 18) * 12
    return df.index[fora_emprego | fora_banco]


def defeituosas(df: pd.DataFrame) -> pd.Index:
    """Linhas com falha COMPROVADA. Nao confundir com `fora_do_dominio`. Aqui: nenhuma.

    Fora do dominio documentado nao e' sinonimo de defeito. O criterio e' EVIDENCIA:
    defeito de extracao se ESPALHA; fenomeno se CONCENTRA num subgrupo com sentido.
    Remover linha so' com prova, nunca porque a documentacao nao previa o valor.
    """
    return df.index[pd.Series(False, index=df.index)]


def bruto() -> pd.DataFrame:
    """O arquivo exatamente como veio. Unico ponto do projeto que chama `read_csv`.

    Se o CSV ainda nao existe, gera: o dado e' deterministico (semente fixa).
    """
    if not RAW.exists():
        import gerar_dados
        gerar_dados.main()
    return pd.read_csv(RAW)


def carregar(remover_impossiveis: bool = False) -> pd.DataFrame:
    """Base de tudo: o CSV sem as linhas 100% identicas.

    `remover_impossiveis=True` tira apenas as linhas com falha comprovada (`defeituosas`).
    """
    df = bruto().drop_duplicates()
    if remover_impossiveis:
        df = df.drop(defeituosas(df))
    return df


# ---------------------------------------------------------------------------
# 2. As visoes.
# ---------------------------------------------------------------------------
SEMENTE = 42
PROPORCAO = {"treino": 0.60, "calibracao": 0.20, "teste": 0.20}


def conjuntos(semente: int = SEMENTE) -> dict[str, pd.DataFrame]:
    """Treino 60% / calibracao 20% / teste 20%, estratificados pelo alvo.

    USE PARA: tudo que treina, calibra, escolhe limiar ou reporta.

    POR QUE TRES, e nao dois. Calibrar no teste e reportar no mesmo teste e' vazamento, e
    escolher o LIMIAR no teste tambem e', pelo mesmo motivo. Entao: treina no treino,
    calibra e escolhe limiar na calibracao, e o teste so' e' tocado para relatar.

    POR QUE SORTEIO ALEATORIO, e nao corte temporal: nao ha' coluna de data. Se um dia
    aparecer data, esta escolha cai.

    ESTRATIFICADO porque a classe rara tem ~7,7%: sem estratificar, a proporcao de
    positivos oscila entre os conjuntos e as metricas passam a comparar populacoes
    diferentes.
    """
    from sklearn.model_selection import train_test_split

    df = carregar()
    treino, resto = train_test_split(
        df, test_size=PROPORCAO["calibracao"] + PROPORCAO["teste"],
        stratify=df[ALVO], random_state=semente,
    )
    fatia = PROPORCAO["teste"] / (PROPORCAO["calibracao"] + PROPORCAO["teste"])
    calibracao, teste = train_test_split(
        resto, test_size=fatia, stratify=resto[ALVO], random_state=semente,
    )
    return {"treino": treino, "calibracao": calibracao, "teste": teste}


def X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e alvo. Tira `id_solicitacao`: identificador nao e' caracteristica."""
    return df.drop(columns=NAO_FEATURES), df[ALVO]


if __name__ == "__main__":
    print(f"{'visao':<20} {'linhas':>9}")
    print(f"{'bruto':<20} {len(bruto()):>9,}")
    print(f"{'carregar':<20} {len(carregar()):>9,}")
    for nome, parte in conjuntos().items():
        print(f"{'conjuntos[' + nome + ']':<20} {len(parte):>9,}   "
              f"{parte[ALVO].mean():.2%} de positivos")
    print(f"\nlinhas fora do dominio declarado: {len(fora_do_dominio(carregar())):,}")
