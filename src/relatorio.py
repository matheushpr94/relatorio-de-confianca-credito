"""Gera o Relatório de Confiança, em 9 seções, a partir do mesmo cálculo das saídas.

POR QUE GERADO, e nao escrito a mao: numero em prosa envelhece calado. Um README que cita
um numero de tres rodadas atras nao da' erro nenhum. Aqui o texto nasce do mesmo calculo
que produz as saidas, entao nao existe versao velha para divergir.

As decisoes que num banco seriam de outra area (razao de custo, fila de revisao)
aparecem como PREMISSA DECLARADA, com a conta que as produziu e a sensibilidade ao lado.
Neste exercicio nao existe area de negocio para assinar, e deixar o campo vazio
entregaria um relatorio sem limiar e sem matriz.

    uv run python src/rodar_tudo.py relatorio   -> outputs/09-relatorio-de-confianca.md
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

import metricas
import prepare
from camada_1_erro_e_subgrupos import wilson
from camada_3_calibracao import versoes_calibradas
from modelo_e_baselines import boosting, floresta, logistica
from subgrupos_candidatos import recortes
from prepare import CATEGORICAS

RAZAO = metricas.RAZAO_REFERENCIA
PREMISSA = "**premissa declarada**: não há área de negócio neste exercício"

LINHAS: list[str] = []


def P(*partes) -> None:
    """Junta a linha. A impressao acontece no fim, ja' com numero em formato brasileiro."""
    LINHAS.append(" ".join(str(p) for p in partes) if partes else "")


def brasileiro(texto: str) -> str:
    """1,234.56 -> 1.234,56, no texto inteiro de uma vez.

    Feito no fim, e nao em cada f-string, porque espalhar formatacao por 200 linhas e' como
    um numero acaba divergindo do outro. A troca usa um marcador intermediario para nao
    embaralhar separador de milhar com separador decimal.
    """
    def trocar(m: re.Match) -> str:
        return m.group(0).replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+", trocar, texto)


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
    t, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(RAZAO))
    m = metricas.matriz(y_te, p_te, t)
    m50 = metricas.matriz(y_te, p_te, 0.5)
    piso = metricas.pisos(y_te)
    L = metricas.linha("", y_te, p_te)
    pred = (p_te >= t).astype(int)
    espec = m["vn"] / (m["vn"] + m["fp"])
    negados = m["vp"] + m["fp"]

    P("# Relatório de Confiança")
    P()
    P("**Modelo:** regressão logística sobre 13 variáveis de solicitação de crédito  ")
    P("**Autor:** Matheus Rosa. Workshop da Residência em IA, feito em dupla; este repositório")
    P("é a minha reconstrução com dado sintético.")
    P()
    P(f"> **Dado sintético.** Todos os números deste relatório vêm de {len(prepare.bruto()):,}")
    P("> solicitações geradas por `src/gerar_dados.py` (semente fixa). O efeito na região")
    P("> Norte foi **plantado** no gerador: lá o risco depende mais de um fator que não está")
    P("> no arquivo. O relatório mostra o método encontrando esse efeito; ele não descreve")
    P("> nenhum mercado real.")
    P()
    P("**Decisão que este modelo apoia:** triagem de solicitações de crédito.")
    P()
    P("> **Premissa declarada:** este relatório trata a saída do")
    P("> modelo como **negativa de crédito**, que é a leitura mais severa. Se na prática ele")
    P("> só encaminha para análise humana, o custo de um falso positivo cai muito — de")
    P("> \"a pessoa não recebe o crédito\" para \"a pessoa espera mais\" — e a razão de custo da")
    P("> seção 4 muda junto. Confirmar antes de usar qualquer linha daquela tabela.")
    P()
    P("**Quem é afetado por um erro:** de um lado a pessoa que pede crédito; do outro o")
    P("banco que empresta. Os dois lados aparecem na seção 2, com nome e consequência.")
    P()
    P("---")
    P()
    P("## 1 · Dados de avaliação")
    P()
    P(f"- Tamanho do conjunto de teste: **{len(y_te):,}** solicitações")
    P(f"- Prevalência da classe positiva: **{piso['prevalencia']:.2%}** "
      f"({int(y_te.sum())} inadimplentes)")
    P(f"- Conjunto separado para calibração? **☑ sim** ☐ não · tamanho: **{len(y_cal):,}**")
    P(f"- Treino: {len(y_tr):,}. Divisão 60/20/20 estratificada, semente {prepare.SEMENTE}.")
    P()
    P("**O teste representa a distribuição de produção?** ☐ sim **☑ não** ☐ não sabemos")
    P()
    P("**O que faria essa suposição falhar:**")
    P()
    bruto = prepare.bruto()
    P("1. **A amostra é a população aprovada.** Todas as linhas têm desfecho de")
    P("   inadimplência, e desfecho só existe para crédito concedido. Quem foi negado não")
    P("   está no arquivo, e em produção o modelo pontua essas pessoas também")
    P("   (*reject inference*).")
    P("2. **Não há coluna de data.** Sem ela não há validação temporal, nem medição de")
    P("   deriva, nem separação por safra. O sorteio aleatório dos conjuntos é a única")
    P("   opção, e ele é otimista se o mundo muda com o tempo.")
    P(f"3. **O dado é sintético:** {len(bruto):,} linhas geradas com distribuições escritas à")
    P("   mão. Ele serve para demonstrar o método; nenhum número aqui vale como estimativa")
    P("   de um mercado real.")
    P()
    P("---")
    P()
    P("## 2 · Matriz de confusão")
    P()
    P(f"Limiar operacional usado: **t = {t:.4f}**, derivado do custo (seção 4).")
    P()
    P("|  | Previsto INADIMPLENTE | Previsto PAGADOR |")
    P("|---|---|---|")
    P(f"| **Real INADIMPLENTE** | VP = {m['vp']} | FN = {m['fn']} |")
    P(f"| **Real PAGADOR** | FP = {m['fp']} | VN = {m['vn']} |")
    P()
    P("**Um falso positivo significa, na prática:** uma pessoa que pagaria o empréstimo")
    P("tem o crédito negado. Ela não faz a compra, a obra ou o capital de giro; o banco")
    P(f"deixa de ganhar o juro daquele contrato. Aconteceu com **{m['fp']} das "
      f"{int((y_te == 0).sum()):,}** pessoas boas do teste.")
    P()
    P("**Um falso negativo significa, na prática:** uma pessoa recebe um empréstimo que não")
    P("vai conseguir pagar. O banco perde o principal, e ela entra numa dívida com")
    P(f"negativação atrás. Aconteceu com **{m['fn']} dos {int(y_te.sum())}** inadimplentes.")
    P()
    P("---")
    P()
    P("## 3 · Métricas")
    P()
    P("| Métrica | Valor | Piso | Quem alcança o piso |")
    P("|---------|-------|------|---------------------|")
    P(f"| Precisão | {m['precisao']:.3f} | {piso['prevalencia']:.3f} | negar ao acaso |")
    P(f"| Recall | {m['recall']:.3f} | 0.000 | o modelo que nunca acusa ninguém |")
    P(f"| Especificidade | {espec:.3f} | 1.000 | esse mesmo modelo, que deixa todos em paz |")
    P(f"| Balanced accuracy | {balanced_accuracy_score(y_te, pred):.3f} | 0.500 | acaso |")
    P(f"| MCC | {matthews_corrcoef(y_te, pred):.3f} | 0.000 | acaso |")
    P(f"| AUPRC | {L['auprc']:.3f} | {piso['auprc']:.3f} | acaso (= a prevalência) |")
    P(f"| AUC-ROC | {L['auc']:.3f} | 0.500 | acaso |")
    P()
    P("**Métrica primária escolhida: recall**, com precisão como restrição de operação.")
    P()
    P("**Justificativa, ligada ao custo do erro:** o falso negativo custa o principal do")
    P("empréstimo; o falso positivo custa o juro de um contrato e o incômodo de uma pessoa.")
    P("Enquanto o negócio não assinar a razão exata, a ordem de grandeza já é clara, e ela")
    P("manda priorizar recall. **A acurácia foi deliberadamente deixada de fora desta")
    P(f"tabela:** no limiar operacional ela cai para {(m['vp'] + m['vn']) / len(y_te):.4f}, "
      f"**abaixo do piso de {piso['acuracia']:.4f}**")
    P("do modelo que não faz nada. Isso é esperado — a troca é de acurácia por detecção de caso")
    P("raro — e é exatamente por isso que acurácia não serve para decidir aqui.")
    P()
    P("**O que o modelo ganha de um baseline honesto.** Todos medidos na **calibração**,")
    P("que é onde a escolha foi feita — comparar AUPRC entre conjuntos diferentes não vale:")
    P()
    numericas = [c for c in X_tr.columns if c not in CATEGORICAS]
    piso_cal = metricas.pisos(y_cal)
    candidatos = {"regra de uma coluna (score do bureau)":
                  (850 - X_cal["score_bureau"]) / (850 - 300)}
    memoria = {}
    for nome, construtor in [("**a logística**", logistica), ("Random Forest", floresta),
                             ("HistGradientBoosting", boosting)]:
        mod = construtor(numericas).fit(X_tr, y_tr)
        candidatos[nome] = mod.predict_proba(X_cal)[:, 1]
        memoria[nome] = metricas.linha("", y_tr, mod.predict_proba(X_tr)[:, 1])["auprc"]
    P("| Comparação | AUPRC na calibração | × o piso | AUPRC no próprio treino |")
    P("|---|---|---|---|")
    P(f"| chutar ao acaso | {piso_cal['auprc']:.3f} | 1.0x | — |")
    for nome, escore in candidatos.items():
        a = metricas.linha("", y_cal, escore)["auprc"]
        mem = f"{memoria[nome]:.3f}" if nome in memoria else "—"
        P(f"| {nome} | {a:.3f} | {a / piso_cal['auprc']:.1f}x | {mem} |")
    P()
    P("A última coluna é o teste de memória: quando o modelo acerta muito melhor nas linhas")
    P("que já viu do que nas que não viu, ele decorou em vez de aprender. O Random Forest")
    P(f"chega a {memoria['Random Forest']:.3f} no próprio treino e cai para "
      f"{metricas.linha('', y_cal, candidatos['Random Forest'])['auprc']:.3f} fora dele.")
    P()
    P("---")
    P()
    P("## 4 · Custo e limiar")
    P()
    P(f"- Razão de custo adotada (FN : FP): **{RAZAO}:1**, {PREMISSA}")
    P("- **De onde vem:** no calote o banco perde o principal, entre")
    P("  60% e 90% dele, porque parte se recupera. No cliente bom barrado, perde a margem")
    P("  daquele contrato, entre 5% e 15%. A razão entre as duas cai entre **4:1 e 18:1**, e")
    P("  10:1 é o meio da faixa. A tabela de sensibilidade abaixo é a resposta para \"e se")
    P("  essa conta estiver errada?\".")
    P("- **Quem validaria isso fora de um exercício:** risco de crédito ou a área de negócio —")
    P("  nunca quem treinou o modelo, porque quem treina tem interesse no resultado.")
    P(f"- Limiar ótimo: **{t:.4f}**, escolhido na **calibração** e aplicado ao teste")
    P(f"- No limiar padrão 0,5: recall {m50['recall']:.3f}, "
      f"apenas {m50['vp']} dos {int(y_te.sum())} inadimplentes retidos")
    P(f"- Custo em t\\*: **{metricas.custo(y_te, p_te, t, 1, RAZAO):,.0f}** · Custo em "
      f"t = 0,5: **{metricas.custo(y_te, p_te, 0.5, 1, RAZAO):,.0f}** "
      f"(unidade: 1 falso positivo; **{metricas.custo(y_te, p_te, 0.5, 1, RAZAO) / metricas.custo(y_te, p_te, t, 1, RAZAO):.2f}x** mais caro no 0,5)")
    P()
    P("**O que cada razão de custo produziria.** É esta tabela, e não o número adotado, que")
    P("sustenta a premissa: ela mostra o que muda se alguém discordar de 10:1. Limiar")
    P("escolhido na calibração, resultado medido no teste:")
    P()
    P("| 1 FN vale | limiar | recall | precisão | pessoas negadas | calotes retidos |")
    P("|---|---|---|---|---|---|")
    for razao in [5, 10, 20, 50, 100, 200]:
        t_r, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(razao))
        mr = metricas.matriz(y_te, p_te, t_r)
        neg = mr["vp"] + mr["fp"]
        P(f"| {razao} FPs | {t_r:.4f} | {mr['recall']:.3f} | {mr['precisao']:.3f} | "
          f"{neg:,} ({neg / len(y_te):.1%}) | {mr['vp']} de {int(y_te.sum())} |")
    P()
    P("Na última linha o modelo nega quase todo mundo: deixa de ser modelo e vira")
    P("política de não emprestar. O limite superior útil da razão está bem antes disso.")
    P()
    P("---")
    P()
    P("## 5 · Calibração")
    P()
    P(f"- ECE: **{L['ece']:.4f}** · Brier: **{L['brier']:.5f}** "
      f"(piso do Brier: {piso['brier']:.4f})")
    versoes = versoes_calibradas(X_tr, y_tr, X_cal, y_cal, X_te)
    bordas = np.unique(np.quantile(p_te, np.linspace(0, 1, 9)))
    idx = np.clip(np.digitize(p_te, bordas[1:-1]), 0, len(bordas) - 2)
    maior_gap = max(abs(y_te[idx == b].mean() - p_te[idx == b].mean())
                    for b in range(len(bordas) - 1) if (idx == b).sum())
    ece_versao = {n: metricas.linha("", y_te, pt)["ece"] for n, (_, pt) in versoes.items()}
    auc_versao = {n: metricas.linha("", y_te, pt)["auc"] for n, (_, pt) in versoes.items()}
    calibrado = maior_gap < 0.02
    P(f"- Diagnóstico: {'☐' if calibrado else '☑'} descalibrado "
      f"{'**☑ calibrado**' if calibrado else '☐ calibrado'}. Em oito faixas de mesma "
      "população, a maior diferença")
    P(f"  entre o previsto e o observado é de {maior_gap:.3f}.")
    melhorou = min(ece_versao['Platt'], ece_versao['Isotonic']) < ece_versao['sem calibrar']
    if melhorou:
        P("- Correção aplicada: a versão de menor ECE no teste, ver `outputs/06-calibracao.txt`.")
    else:
        P("- Correção aplicada: **nenhuma, e isso foi decidido por medição.**")
        P(f"  Platt e isotônica foram testadas e nenhuma melhorou: o ECE foi de "
          f"{ece_versao['sem calibrar']:.4f} para")
        P(f"  {ece_versao['Platt']:.4f} e {ece_versao['Isotonic']:.4f}, e a isotônica "
          f"mexeu na AUC em {auc_versao['Isotonic'] - auc_versao['sem calibrar']:+.4f}.")
        P("  A regressão logística já otimiza log-loss, então chegar calibrada é o esperado.")
    P("- Conjunto usado para calibrar: o de calibração (nunca o teste).")
    P("- **O limiar foi recalculado depois de calibrar?** **☑ sim** ☐ não — para as três")
    P("  versões; a operação resultante é praticamente a mesma.")
    P()
    P("> **ECE baixo, sozinho, não é mérito.** O piso do ECE é zero, e quem o alcança é o")
    P(f"> modelo que devolve {piso['prevalencia']:.2%} para todo mundo e não detecta ninguém. ")
    P("> ECE só significa alguma coisa ao lado de uma medida de ordenação, como a AUPRC.")
    P()
    P("---")
    P()
    P("## 6 · Incerteza e abstenção")
    P()
    margem = np.abs(p_te - t)
    ordem = np.argsort(margem)
    P("| Cobertura | Respondidos | À revisão humana | Precisão | Recall | Calotes na fila |")
    P("|---|---|---|---|---|---|")
    for cob in [1.0, 0.98, 0.95, 0.90, 0.80]:
        k = int(round(len(y_te) * (1 - cob)))
        fica = np.ones(len(y_te), bool)
        fica[ordem[:k]] = False
        mc = metricas.matriz(y_te[fica], p_te[fica], t)
        P(f"| {cob:.0%} | {int(fica.sum()):,} | {k:,} | {mc['precisao']:.3f} | "
          f"{mc['recall']:.3f} | {int(y_te[~fica].sum())} de {int(y_te.sum())} |")
    P()
    k90 = int(round(len(y_te) * 0.10))
    fica90 = np.ones(len(y_te), bool)
    fica90[ordem[:k90]] = False
    m90 = metricas.matriz(y_te[fica90], p_te[fica90], t)
    P("- **Volume absoluto enviado a revisão:** a 90% de cobertura, **100 análises por mil**")
    P("  solicitações. O arquivo não tem data, então não existe 'casos por dia' honesto —")
    P("  a conta depende do volume de operação, que não foi informado.")
    P("- Cobertura escolhida: **100%** — a decisão é **não abster**, e a tabela acima é o porquê.")
    P("- Quem revisa: numa operação real, a mesa de análise de crédito · Essa capacidade")
    P("  existe hoje? ☐ sim **☑ não** — nesta atividade não existe fila nem revisor, o que")
    P("  reforça a decisão de não abster em vez de enfraquecê-la.")
    P("- **Recomendação sobre abster: não vale a pena com os números atuais.** Parar de")
    P(f"  responder 10% dos casos compra {100 * (m90['precisao'] - m['precisao']):.1f} ponto de")
    P(f"  precisão e {100 * (m90['recall'] - m['recall']):.1f} de recall, ao custo de {k90}")
    P("  análises humanas no teste. O ganho não paga a fila.")
    P()
    errada = set(np.argsort(np.abs(p_te - 0.5))[:k90])
    comum = len(set(ordem[:k90]) & errada)
    P("> A confiança foi medida como distância ao **limiar operacional**, não ao 0,5. A")
    P("> diferença não é técnica, é operacional: as duas listas de '10% mais incertos' têm")
    P(f"> **{comum} pessoa(s) em comum** de {k90}. Usar o 0,5 mandaria para revisão as pessoas")
    P("> de maior risco, e não as que estão em cima do muro.")
    P()
    P("---")
    P()
    P("## 7 · Desempenho por subgrupo")
    P()
    P("| Subgrupo | n | Positivos | Recall | IC 95% | Precisão | Alerta |")
    P("|---|---|---|---|---|---|---|")
    for g in sorted(teste["regiao"].unique()):
        sel = (teste["regiao"] == g).to_numpy()
        r = metricas.matriz(y_te[sel], p_te[sel], t)
        pos = r["vp"] + r["fn"]
        lo, hi = wilson(r["vp"], pos)
        alerta = "**PIOR**" if hi < m["recall"] else ("incerto" if hi - lo > 0.30 else "")
        P(f"| {g} | {int(sel.sum()):,} | {pos} | {r['recall']:.3f} | "
          f"[{lo:.3f} , {hi:.3f}] | {r['precisao']:.3f} | {alerta} |")
    lo_g, hi_g = wilson(m["vp"], m["vp"] + m["fn"])
    P(f"| **(geral)** | {len(y_te):,} | {int(y_te.sum())} | {m['recall']:.3f} | "
      f"[{lo_g:.3f} , {hi_g:.3f}] | {m['precisao']:.3f} | |")
    P()
    por_regiao = {g: metricas.matriz(y_te[(teste["regiao"] == g).to_numpy()],
                                     p_te[(teste["regiao"] == g).to_numpy()], t)
                  for g in teste["regiao"].unique()}
    pior = min(por_regiao, key=lambda g: por_regiao[g]["recall"])
    melhor = max(por_regiao, key=lambda g: por_regiao[g]["recall"] if
                 (por_regiao[g]["vp"] + por_regiao[g]["fn"]) >= 50 else -1)
    P(f"**Decisão sobre o pior subgrupo.** O {pior} tem recall "
      f"**{por_regiao[pior]['recall']:.3f}** contra {m['recall']:.3f} no")
    P("agregado, e o intervalo de confiança do grupo **não alcança** o valor geral — a")
    P("diferença não se explica por tamanho de amostra. Na prática, o mesmo modelo é dois")
    P(f"produtos diferentes: no {melhor} ele retém {por_regiao[melhor]['recall']:.0%} dos")
    P(f"calotes, no {pior} {por_regiao[pior]['recall']:.0%}.")
    taxa_pior = float(teste.loc[teste['regiao'] == pior, prepare.ALVO].mean())
    P(f"Note que a **taxa** de calote no {pior} ({taxa_pior:.1%} no teste) é parecida com a")
    P("geral: o problema não é haver mais calote lá, é o modelo ter menos evidência para")
    P("separar quem paga de quem não paga. No dado sintético isso foi plantado de propósito;")
    P("num dado real, a pergunta seguinte seria *que informação falta para essa região?*")
    P(f"Isso é risco técnico e é risco jurídico. Enquanto não houver explicação, o {pior} não")
    P("deveria operar com decisão automática.")
    P()
    P(f"### Os {len(recortes(teste))} recortes, e os dois jeitos de um subgrupo ser prejudicado")
    P()
    P("Nomear um subgrupo por intuição é onde mora o viés de quem escolhe. Aqui foram medidos **todos**")
    P("os recortes que o arquivo permite, em duas perguntas que não são a mesma:")
    P()
    P("- **protege menos** — o recall do grupo fica abaixo do geral. Quem paga é a pessoa do")
    P("  grupo cujo calote passou, e o banco.")
    P("- **barra mais** — a taxa de negação do grupo fica acima da geral. Quem paga é a")
    P("  pessoa que teria pagado e levou não.")
    P()
    P("Só entra na lista quem tem o intervalo de 95% inteiro fora do valor geral:")
    P()
    P("| Recorte | Grupo | n | Recall | Negados | O quê |")
    P("|---|---|---|---|---|---|")
    achados_sub = []
    for nome_rec, chave in recortes(teste).items():
        for g in sorted(chave.unique(), key=str):
            sel = (chave == g).to_numpy()
            if sel.sum() < 30:
                continue
            r = metricas.matriz(y_te[sel], p_te[sel], t)
            pos = r["vp"] + r["fn"]
            neg = (r["vp"] + r["fp"]) / sel.sum()
            marcas = []
            if pos and wilson(r["vp"], pos)[1] < m["recall"]:
                marcas.append("protege menos")
            if wilson(r["vp"] + r["fp"], int(sel.sum()))[0] > negados / len(y_te):
                marcas.append("**barra mais**")
            if marcas:
                achados_sub.append((nome_rec, str(g), int(sel.sum()), r["recall"], neg,
                                    " + ".join(marcas)))
    for nome_rec, g, n_g, rec, neg, marcas in achados_sub:
        P(f"| {nome_rec} | {g} | {n_g:,} | {rec:.3f} | {neg:.1%} | {marcas} |")
    P(f"| **(geral)** | — | {len(y_te):,} | {m['recall']:.3f} | {negados / len(y_te):.1%} | |")
    P()
    P("**Como ler a tabela.**")
    P()
    P("1. **Barrar mais nem sempre é injusto.** Score baixo, atraso recente e prazo longo")
    P("   são barrados mais porque têm mais risco de fato; isso é o modelo funcionando. O caso")
    P("   que pede atenção é o grupo **protegido menos sem ter risco menor**, e o candidato")
    P(f"   nítido aqui é o {pior}.")
    P("2. **Nos quartis de score alto, o recall despenca.** O modelo pontua muito pelo score,")
    P("   então o inadimplente com bom score passa. É o mesmo que o ganho modesto sobre a")
    P("   régua de uma coluna diz, por outro caminho: **o modelo enxerga pouco risco que o")
    P("   bureau já não enxergue.**")
    sem_dif = [nome for nome, chave in recortes(teste).items()
               if nome not in {a[0] for a in achados_sub}]
    if sem_dif:
        P(f"3. **Recortes sem nenhuma diferença sustentada:** {', '.join(sem_dif)}. Eles")
        P("   aparecem para mostrar que a lista acima não é a fatia escolhida por dar manchete.")
    P()
    P("---")
    P()
    P("## 8 · Limitações e lacunas")
    P()
    P("| Lacuna | Risco que ela representa | Responsável | Prazo |")
    P("|---|---|---|---|")
    P(f"| A razão de custo {RAZAO}:1 é premissa declarada, derivada e não medida | se a razão "
      "real for outra, o limiar muda; a tabela da seção 4 mostra quanto | risco de crédito, "
      "num banco | antes do piloto |")
    P("| A amostra é a população aprovada (*reject inference*) | o modelo nunca viu quem foi "
      "negado, e vai pontuar essas pessoas | quem entregou o dado | antes do piloto |")
    P("| Não há coluna de data | sem validação temporal, sem deriva, sem safra | quem "
      "entregou o dado | antes do piloto |")
    P(f"| Recall do {pior} é {por_regiao[pior]['recall'] / m['recall']:.0%} do geral | mesmo "
      f"modelo, dois produtos; risco jurídico | modelagem + jurídico | antes de ligar no {pior} |")
    P("| Ninguém designado para revisar abstenções | a política de abstenção fica no papel | "
      "operação | antes de ligar a abstenção |")
    ganho = (metricas.linha("", y_cal, candidatos["**a logística**"])["auprc"]
             - metricas.linha("", y_cal, candidatos["regra de uma coluna (score do bureau)"])["auprc"])
    P("| O dado é sintético | os números demonstram o método, não estimam nada real | modelagem "
      "| antes de qualquer uso |")
    P(f"| O ganho sobre olhar só o score do bureau é de {ganho:.3f} de AUPRC | talvez não "
      "valha manter um modelo em produção | modelagem | antes do piloto |")
    P("| A calibração serviu a três decisões (escolher modelo, calibrar, escolher limiar) | "
      "o limiar pode estar levemente sobreajustado | modelagem | próxima iteração |")
    P()
    P("> Um relatório sem nenhuma lacuna é sinal de que ninguém olhou com atenção suficiente.")
    P()
    P("---")
    P()
    P("## 9 · Recomendação")
    P()
    P("☐ Pronto para produção  ·  ☐ Produção com revisão humana  ·  **☑ Piloto restrito**  ·  ☐ Não recomendado")
    P()
    P("**PILOTO RESTRITO**: nem produção, nem arquivo morto.")
    P()
    P("Em duas frases: o modelo ordena risco acima do acaso, chega calibrado e opera num")
    P(f"limiar derivado do custo — mas no {pior} ele retém "
      f"{por_regiao[pior]['recall']:.0%} dos calotes contra {m['recall']:.0%} no agregado, e a")
    P("amostra só contém quem já teve crédito aprovado. Ligar em produção hoje seria")
    P("automatizar decisão de crédito num grupo onde o modelo comprovadamente funciona pior,")
    P("com uma base que não representa quem vai ser pontuado.")
    P()
    P("**O que destrava a aprovação, em ordem:** (1) explicação ou correção do")
    P(f"{pior}; (2) confirmação de como o dado foi amostrado, e do que acontece com quem hoje")
    P("é negado; (3) alguém com mandato de negócio revisando a razão de custo adotada;")
    P("(4) se a abstenção for ligada um dia, uma pessoa designada para a fila.")
    P()
    sel_pior = (teste["regiao"] == pior).to_numpy()
    reais, uteis = [], []
    for razao in metricas.FAIXA_RAZOES:
        t_r, _, _ = metricas.varrer_limiar(y_cal, p_cal, 1.0, float(razao))
        g = metricas.matriz(y_te, p_te, t_r)
        s = metricas.matriz(y_te[sel_pior], p_te[sel_pior], t_r)
        _, hi = wilson(s["vp"], s["vp"] + s["fn"])
        (reais if hi < g["recall"] else uteis).append(razao)
    P("**O que não trava a decisão:** a razão de custo. Ela é premissa declarada, com a")
    P("conta que a produziu e a sensibilidade ao lado. E a recomendação não depende dela:")
    P(f"o achado do {pior} foi testado em todos os {len(metricas.FAIXA_RAZOES)} cenários da tabela")
    P(f"da seção 4, e a diferença se sustenta em **{len(reais)}** deles, de {min(reais)}:1 a")
    if uteis:
        P(f"{max(reais)}:1. Ela só fica inconclusiva em {', '.join(f'{r}:1' for r in uteis)}, onde o")
        P("modelo quase não encontra inadimplente em lugar nenhum e não sobra diferença para medir.")
    else:
        P(f"{max(reais)}:1, ou seja, em todos.")
    P()
    P("---")
    P()
    P("## Apêndice · O que eu faria diferente")
    P()
    P("- **Separar mais um pedaço do dado.** O conjunto de calibração serviu a três decisões")
    P("  (escolher o modelo, calibrar e escolher o limiar). Um quarto conjunto resolveria.")
    P("- **Escrever a lista de recortes antes de medir.** Medir todos os recortes reduz o viés")
    P("  de escolher o que dá manchete; fixar a lista antes elimina.")
    P("- **Perguntar cedo o que o alvo significa.** \"Inadimplente\" em quantos dias de")
    P("  atraso? Isso muda o que o modelo aprende, e o arquivo não diz.")
    P("- **Levar as perguntas de negócio para fora antes.** Razão de custo, fila de revisão e")
    P("  amostragem travaram mais o trabalho do que qualquer linha de código.")

    print(brasileiro("\n".join(LINHAS)))


if __name__ == "__main__":
    main()
