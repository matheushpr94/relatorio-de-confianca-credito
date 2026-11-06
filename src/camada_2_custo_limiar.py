"""Qual erro dói mais, e quanto? O limiar deixa de ser opinião.

Camada 2. A razao de custo FN:FP e' decisao de NEGOCIO, e neste exercicio nao existe
area de negocio para assinar. Entao uma razao e' ADOTADA, mostra a conta que a
produziu (ver `metricas.RAZAO_REFERENCIA`) e varre a faixa inteira ao lado, para qualquer
pessoa ver o que muda se discordar do numero.

Deixar a razao em aberto seria mais confortavel e menos util: sem ela nao existe limiar,
e sem limiar o relatorio nao tem matriz de confusao nenhuma para mostrar.

Escolher o limiar no TESTE e relatar o ganho no mesmo TESTE e' vazamento, pelo mesmo
motivo que calibrar no teste e'. Aqui o limiar sai da CALIBRACAO e o ganho e' medido no
TESTE, e a ultima secao mede quanto o atalho inflaria o resultado.
"""
from __future__ import annotations

import numpy as np

import metricas
import prepare
from modelo_e_baselines import logistica
from prepare import CATEGORICAS

# A faixa varrida, e a referencia adotada, vivem em metricas.py — com a conta que
# justifica o 10:1 escrita la'.
RAZOES = metricas.FAIXA_RAZOES
C_FP = 1.0   # unidade: custo de negar credito a quem pagaria


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

    titulo("O LIMIAR 0,5 NAO FOI ESCOLHIDO POR NINGUEM")
    m = metricas.matriz(y_te, p_te, 0.5)
    piso = metricas.pisos(y_te)
    print(f"   no teste, limiar 0,5:  VP {m['vp']}  FN {m['fn']}  FP {m['fp']}  VN {m['vn']}")
    print(f"   recall {m['recall']:.3f} | precisao {m['precisao']:.3f}")
    print(f"   acuracia {(m['vp'] + m['vn']) / len(y_te):.4f} contra piso {piso['acuracia']:.4f} "
          f"do modelo que nunca acusa ninguem")
    print(f"\n   {m['fn']} dos {int(y_te.sum())} inadimplentes passam. O default da biblioteca")
    print("   entrega, aqui, quase o modelo que nunca acusa ninguem.")

    titulo("CADA RAZAO DE CUSTO PRODUZ UM LIMIAR — escolha a linha, nao o numero")
    print("Razao = quantos falsos positivos valem um falso negativo. Em credito:")
    print("  FP = negar credito a quem pagaria (a pessoa perde o credito, o banco o juro)")
    print("  FN = conceder a quem nao paga (o banco perde o principal, a pessoa a divida)")
    print("\nLimiar escolhido na CALIBRACAO, resultado medido no TESTE:\n")
    cab = (f"{'FN:FP':>7}{'t* formula':>12}{'t* varrido':>12}{'recall':>9}{'precisao':>10}"
           f"{'negados':>9}{'% negados':>11}{'calotes retidos':>17}")
    print(cab)
    print("-" * len(cab))
    for razao in RAZOES:
        c_fn = C_FP * razao
        t_formula = C_FP / (C_FP + c_fn)
        t_star, _, _ = metricas.varrer_limiar(y_cal, p_cal, C_FP, c_fn)
        m = metricas.matriz(y_te, p_te, t_star)
        negados = m["vp"] + m["fp"]
        print(f"{razao:>6}x{t_formula:>12.4f}{t_star:>12.4f}{m['recall']:>9.3f}"
              f"{m['precisao']:>10.3f}{negados:>9,}{negados / len(y_te):>10.1%}"
              f"{m['vp']:>10} de {int(y_te.sum())}")

    print("\n   A formula fechada e o limiar varrido discordam porque a formula supoe")
    print("   probabilidade perfeitamente calibrada e custo linear; na amostra finita, os")
    print("   dois se afastam. Onde discordam, manda o varrido.")

    titulo("O CUSTO DE ERRAR O LIMIAR E' ASSIMETRICO")
    c_fn = C_FP * metricas.RAZAO_REFERENCIA
    t_star, limiares, custos = metricas.varrer_limiar(y_cal, p_cal, C_FP, c_fn)
    base = metricas.custo(y_te, p_te, t_star, C_FP, c_fn)
    print(f"   cenario de referencia adotado: 1 FN = {metricas.RAZAO_REFERENCIA} FPs, t* = {t_star:.4f}\n")
    print(f"   {'limiar':>10}{'custo no teste':>18}{'x o otimo':>12}")
    for t in [t_star / 4, t_star / 2, t_star, t_star * 2, t_star * 4, 0.5]:
        if t > 0.999:
            continue
        c = metricas.custo(y_te, p_te, t, C_FP, c_fn)
        marca = "  <- otimo" if abs(t - t_star) < 1e-9 else ("  <- default" if t == 0.5 else "")
        print(f"   {t:>10.4f}{c:>18,.0f}{c / base:>11.2f}x{marca}")
    c50 = metricas.custo(y_te, p_te, 0.5, C_FP, c_fn)
    print(f"\n   Operar no 0,5 custa {c50 / base:.2f}x o otimo. A curva e' rasa perto do otimo")
    print("   e sobe nas pontas: errar o limiar por um fator 2 custa pouco, errar por uma")
    print("   ordem de grandeza custa caro. A figura docs/figuras/custo_limiar.png mostra.")

    titulo("O PRECO DE ESCOLHER E RELATAR NO MESMO CONJUNTO")
    print("Quanto o numero infla quando o limiar e' escolhido e medido na mesma amostra.")
    print("Nao e' erro de conta: e' otimismo, e da' para medir.\n")
    print(f"   {'razao':>7}{'limiar':>9}{'recall DENTRO':>16}{'recall FORA':>14}{'otimismo':>11}")
    for razao in RAZOES:
        t, _, _ = metricas.varrer_limiar(y_cal, p_cal, C_FP, C_FP * razao)
        dentro = metricas.matriz(y_cal, p_cal, t)["recall"]
        fora = metricas.matriz(y_te, p_te, t)["recall"]
        print(f"   {razao:>6}x{t:>9.3f}{dentro:>16.4f}{fora:>14.4f}{dentro - fora:>+11.4f}")
    print("\n   O numero de dentro nao e' estimativa honesta do numero de fora: a diferenca")
    print("   muda de tamanho conforme o cenario, e as duas colunas nunca devem ser")
    print("   apresentadas como se fossem a mesma coisa.")

    titulo("DE QUEM SERIA ESSA DECISAO, FORA DE UM EXERCICIO")
    print("   Num banco: risco de credito, ou a area de negocio — nunca quem treinou o")
    print("   modelo, porque quem treina tem interesse no resultado. A questao e' de")
    print("   PAPEL, nao de burocracia.")
    print()
    print(f"   Aqui nao ha' essa pessoa, entao a razao adotada e' {metricas.RAZAO_REFERENCIA}:1, e")
    print("   a conta fica declarada: no calote o banco perde o principal (60% a 90% dele, porque")
    print("   parte se recupera); no cliente bom barrado perde a margem do contrato (5% a")
    print("   15%). A razao entre as duas cai entre 4:1 e 18:1; 10:1 e' o meio da faixa.")
    print()
    print("   O que isso muda na leitura do relatorio: a linha de referencia e' PREMISSA")
    print("   declarada, e a tabela acima e' a resposta para 'e se a premissa estiver")
    print("   errada?'. Premissa declarada com sensibilidade ao lado nao e' chute.")


if __name__ == "__main__":
    main()
