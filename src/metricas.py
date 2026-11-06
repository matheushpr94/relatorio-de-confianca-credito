"""As medidas das quatro camadas, num lugar so'.

Nao e' script de analise: nao imprime nem grava. Existe para que ECE, piso e varredura de
custo tenham UMA definicao no projeto inteiro — dois scripts com ECE de bins diferentes
produzem dois numeros que ninguem consegue comparar depois.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)


# Razao de custo FN:FP adotada como CENARIO DE REFERENCIA. Fica aqui, num lugar so',
# porque quatro scripts dependem dela — dois valores diferentes em dois scripts
# produziriam duas operacoes diferentes com o mesmo nome.
#
# DE ONDE VEM O 10. Nao ha' custo real informado, e neste exercicio ninguem de fora vai
# assinar. Entao o valor e' assumido, com a conta declarada. Num calote o
# banco perde o PRINCIPAL — entre 60% e 90% dele, porque parte se recupera. Num cliente
# bom barrado, perde a MARGEM daquele contrato — entre 5% e 15%. A razao entre as duas
# cai entre 4:1 e 18:1, e 10:1 e' o meio dessa faixa.
#
# O numero e' premissa, nao medida. Por isso nenhum script trata ele como fixo: todos
# varrem a faixa inteira ao lado, e a tabela de sensibilidade e' parte do entregavel.
RAZAO_REFERENCIA = 10
FAIXA_RAZOES = [1, 2, 5, 10, 20, 50, 100, 200]


def ece(y, p, n_bins: int = 10) -> float:
    """Erro de calibracao esperado, com bins de largura fixa — a definicao mais comum.

    CUIDADO, e vale escrever no relatorio: o piso do ECE e' ZERO, e quem o alcanca e' o
    modelo que devolve a prevalencia para todo mundo. ECE baixo sozinho nao e' merito;
    so' significa alguma coisa ao lado de uma medida de discriminacao (AUPRC, recall).

    Com classe rara, bins de largura fixa deixam quase tudo no primeiro bin. Por isso
    `ece_quantil` existe ao lado: se os dois discordarem muito, a conclusao depende do
    bin e nao do modelo, e isso e' informacao.
    """
    y, p = np.asarray(y), np.asarray(p)
    bordas = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bordas[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def ece_quantil(y, p, n_bins: int = 10) -> float:
    """ECE com bins de mesma POPULACAO, nao de mesma largura."""
    y, p = np.asarray(y), np.asarray(p)
    bordas = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, bordas[1:-1]), 0, len(bordas) - 2)
    e = 0.0
    for b in range(len(bordas) - 1):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def pisos(y) -> dict[str, float]:
    """O piso de cada metrica neste conjunto. Numero sem piso ao lado nao diz nada."""
    p = float(np.mean(y))
    return {"prevalencia": p, "acuracia": 1 - p, "auprc": p, "auc": 0.5,
            "brier": p * (1 - p), "ece": 0.0}


def linha(nome: str, y, p) -> dict:
    """Uma linha da tabela de comparacao: ordenacao e calibracao, lado a lado."""
    return {
        "modelo": nome,
        "auc": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "ece": ece(y, p),
        "ece_q": ece_quantil(y, p),
    }


def custo(y, p, limiar: float, c_fp: float, c_fn: float) -> float:
    """Custo total de operar neste limiar. Sem ganho no verdadeiro positivo.

    A formula fechada t* = C_FP/(C_FP+C_FN) so' vale nesta versao sem ganho e com
    probabilidade perfeitamente calibrada. Se o verdadeiro positivo tiver ganho, o otimo
    desloca. Por isso a varredura empirica manda, e a formula fica como referencia.
    """
    y, p = np.asarray(y), np.asarray(p)
    fp = int(((p >= limiar) & (y == 0)).sum())
    fn = int(((p < limiar) & (y == 1)).sum())
    return c_fp * fp + c_fn * fn


def varrer_limiar(y, p, c_fp: float, c_fn: float, n: int = 999) -> tuple[float, np.ndarray, np.ndarray]:
    """Devolve (limiar de menor custo, limiares, custos).

    O limiar tem que ser escolhido FORA do conjunto em que se reporta o ganho. Escolher no
    teste e relatar no teste e' o mesmo vazamento de calibrar e relatar no mesmo conjunto.
    """
    limiares = np.linspace(0.001, 0.999, n)
    custos = np.array([custo(y, p, t, c_fp, c_fn) for t in limiares])
    return float(limiares[int(custos.argmin())]), limiares, custos


def matriz(y, p, limiar: float) -> dict:
    """VN, FP, FN, VP mais precisao e recall naquele limiar."""
    pred = (np.asarray(p) >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp),
            "precisao": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0)}
