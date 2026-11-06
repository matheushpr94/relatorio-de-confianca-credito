"""Selftest do ESQUEMA: prova que ele PEGA o que deveria pegar.

Teste que so' passa em dado bom nao prova nada — precisa FALHAR quando tem que falhar.
Este arquivo existe para que uma regressao no `prepare.py` apareca como erro em vez de
virar numero errado num relatorio.

    uv run python src/selftest_prepare.py    -> outputs/01-selftest.txt

Codigo de saida 0 se todos os casos passarem, 1 se algum falhar.
"""
import sys

import prepare

falhas = 0


def caso(nome: str, esperado, obtido) -> None:
    global falhas
    ok = esperado == obtido
    if not ok:
        falhas += 1
    print(f"  [{'ok ' if ok else 'FALHOU'}] {nome:52} esperado {esperado!s:<14} obtido {obtido}")


def main() -> int:
    limpo = prepare.carregar(remover_impossiveis=True)

    print("=== 1. defeito e 'fora do dominio documentado' sao coisas DIFERENTES ===")
    print("    o filtro tira falha comprovada, nao tudo que a documentacao proibe")
    caso("nenhum defeito sobra apos a remocao", 0, len(prepare.defeituosas(limpo)))
    # O gerador sintetico respeita o dominio inteiro, inclusive a coerencia entre colunas.
    # Se um dia este numero passar de zero, o gerador quebrou o contrato do schema.
    caso("nenhuma linha fora do dominio", 0, len(prepare.fora_do_dominio(limpo)))
    caso("nenhuma linha incoerente", 0, len(prepare.incoerentes(limpo)))

    print("\n=== 1b. a incoerencia entre colunas e' pega pelo schema ===")
    sujo = limpo.head(500).copy()
    jovem = sujo.index[sujo["idade"] < 30][0]
    sujo.loc[jovem, "tempo_relacionamento_banco_meses"] = 600
    caso("50 anos de banco aos <30 anos de idade", 1, len(prepare.fora_do_dominio(sujo)))
    caso("e incoerentes() aponta a mesma linha", [int(jovem)], [int(i) for i in prepare.incoerentes(sujo)])

    print("\n=== 2. valor fora do dominio e' pego, coluna por coluna ===")
    # A amostra pode conter linhas fora do dominio, entao o invariante nao e' "da' 1" —
    # e' "AUMENTA em exatamente 1". E a linha suja tem que ser uma que ainda esteja
    # dentro do dominio, senao o contador nao sobe e o teste passa por engano.
    provas = [
        ("idade", 17),                  # abaixo do minimo legal
        ("score_bureau", 1000),         # faixa do Serasa, nao a deste arquivo
        ("num_parcelas", 18),           # prazo que o produto nao oferece
        ("canal", "WhatsApp"),          # canal inexistente
        ("possui_imovel", 2),           # binaria com terceiro valor
        ("renda_mensal", 0.0),          # renda zero
        ("divida_renda", -0.5),         # comprometimento negativo
        ("inadimplente", 2),            # alvo fora de {0, 1}
    ]
    amostra = limpo.head(500)
    ja_fora = set(prepare.fora_do_dominio(amostra))
    limpa = next(i for i, idx in enumerate(amostra.index) if idx not in ja_fora)
    base = len(ja_fora)
    print(f"    a amostra de 500 ja' tem {base} linha(s) fora do dominio; esperado = base + 1")
    print(f"    sujando a linha {limpa} da amostra, que esta' dentro do dominio")
    for coluna, valor in provas:
        sujo = amostra.copy()
        sujo.iloc[limpa, sujo.columns.get_loc(coluna)] = valor
        caso(f"{coluna} = {valor}", base + 1, len(prepare.fora_do_dominio(sujo)))

    print("\n=== 3. nulo indevido em coluna nao-nulavel ===")
    sujo = amostra.copy()
    sujo.iloc[limpa, sujo.columns.get_loc("score_bureau")] = None
    caso("score_bureau nulo", base + 1, len(prepare.fora_do_dominio(sujo)))

    print("\n=== 3b. id duplicado e' pego ===")
    # Copiar o id de OUTRA linha. Copiar o da propria linha nao muda nada, e o teste
    # passaria a testar o vazio.
    outra = 1 if limpa == 0 else 0
    sujo = amostra.copy()
    sujo.iloc[limpa, sujo.columns.get_loc("id_solicitacao")] = int(sujo["id_solicitacao"].iloc[outra])
    caso("id_solicitacao repetido", True, len(prepare.fora_do_dominio(sujo)) > base)

    print("\n=== 3c. os tres conjuntos nao se tocam, e mantem a prevalencia ===")
    partes = prepare.conjuntos()
    ids = {n: set(p["id_solicitacao"]) for n, p in partes.items()}
    caso("treino x teste sem interseccao", 0, len(ids["treino"] & ids["teste"]))
    caso("calibracao x teste sem interseccao", 0, len(ids["calibracao"] & ids["teste"]))
    caso("os tres somam o arquivo inteiro", len(limpo),
         sum(len(p) for p in partes.values()))
    # Igual ate' o arredondamento de um positivo: com 1.368 positivos em 18.000, a
    # divisao 60/20/20 nao e' exata, e a diferenca maxima esperada e' de ~0,03 ponto.
    prevalencias = [p[prepare.ALVO].mean() for p in partes.values()]
    caso("prevalencia igual nos tres (diferenca < 0,1 ponto)", True,
         max(prevalencias) - min(prevalencias) < 0.001)

    print("\n=== 4. falha ESTRUTURAL levanta, nao devolve indice vazio ===")
    print("    o caso perigoso: sem a guarda, a tabela errada passaria batido")
    sujo = amostra.rename(columns={amostra.columns[0]: "coluna_renomeada"})
    try:
        resto = prepare.fora_do_dominio(sujo)
        caso("coluna renomeada deveria levantar", "ValueError", f"devolveu {len(resto)} indices")
    except ValueError as erro:
        caso("coluna renomeada levanta ValueError", "ValueError", "ValueError")
        print(f"       {str(erro).splitlines()[0]}")

    print(f"\n=== {'TODOS OS CASOS PASSARAM' if not falhas else f'{falhas} CASO(S) FALHARAM'} ===")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
