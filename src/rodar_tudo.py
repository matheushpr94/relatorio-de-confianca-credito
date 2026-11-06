"""Roda todos os scripts do projeto e regrava `outputs/`.

O MAPA abaixo e' a correspondencia entre script e arquivo de saida, ESCRITA. Sem
ele ela vive na cabeca de quem redireciona na mao, e saida acaba orfa de script.
Script novo entra no MAPA junto com o arquivo de saida.

    uv run python src/rodar_tudo.py            # roda tudo e regrava outputs/
    uv run python src/rodar_tudo.py --conferir # roda e SO' avisa o que mudaria
    uv run python src/rodar_tudo.py relatorio  # roda so' o que casar com o termo

Depois de rodar, `git diff outputs/` mostra exatamente o que mudou de numero.

Forca UTF-8 no processo filho, para console sem UTF-8 nao derrubar script no meio.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
OUTPUTS = RAIZ / "outputs"

# script (sem .py) -> arquivo em outputs/
MAPA = {
    "gerar_dados":         "00-dados-sinteticos.txt",
    "selftest_prepare":    "01-selftest.txt",
    "modelo_e_baselines":  "03-modelo-e-baselines.txt",
    "camada_2_custo_limiar": "04-custo-e-limiar.txt",
    "camada_1_erro_e_subgrupos": "05-erro-e-subgrupos.txt",
    "camada_3_calibracao": "06-calibracao.txt",
    "camada_4_incerteza": "07-incerteza-e-abstencao.txt",
    "subgrupos_candidatos": "08-subgrupos-candidatos.txt",
    "figuras":            "10-figuras.txt",
    "relatorio":          "09-relatorio-de-confianca.md",
    "relatorio_html":     "09-relatorio-de-confianca.html",
    "numeros_do_readme":  "11-numeros-do-readme.txt",
}

ORFAS: list[str] = []

# Modulos sem arquivo proprio de saida.
SEM_SAIDA = ["prepare", "rodar_tudo", "metricas"]


def ambiente() -> dict:
    """UTF-8 no filho, para o console do Windows nao derrubar script no meio."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def rodar(nome: str) -> tuple[bool, str]:
    # cwd = raiz do projeto, igual ao `uv run python src/x.py` digitado a mao.
    r = subprocess.run(
        [sys.executable, f"src/{nome}.py"],
        capture_output=True, text=True, encoding="utf-8", env=ambiente(), cwd=str(RAIZ),
    )
    if r.returncode != 0:
        return False, r.stdout + r.stderr
    # Saida vazia e' falha, nao resultado. Sem esta guarda o gate fica cego: um script
    # que passou a imprimir nada grava o arquivo vazio numa rodada e, na seguinte,
    # `--conferir` compara vazio com vazio e responde "igual".
    if not r.stdout.strip():
        return False, "saida VAZIA — o script rodou sem erro e nao imprimiu nada"
    return True, r.stdout


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("termos", nargs="*", help="roda so' os scripts cujo nome contem o termo")
    p.add_argument("--conferir", action="store_true", help="nao grava; so' diz o que mudaria")
    args = p.parse_args()

    alvos = {k: v for k, v in MAPA.items()
             if not args.termos or any(t in k or t in v for t in args.termos)}
    if not alvos:
        print(f"nenhum script casa com {args.termos}. Disponiveis: {', '.join(MAPA)}")
        return 1

    OUTPUTS.mkdir(exist_ok=True)
    modo = "CONFERINDO" if args.conferir else "RODANDO"
    print(f"{modo} {len(alvos)} script(s)\n")

    mudou, iguais, falhou = [], [], []
    for nome, arquivo in alvos.items():
        ok, saida = rodar(nome)
        if not ok:
            falhou.append(nome)
            print(f"  FALHOU   {nome}")
            print("           " + saida.strip().splitlines()[-1][:100])
            continue
        destino = OUTPUTS / arquivo
        anterior = destino.read_text(encoding="utf-8") if destino.exists() else None
        igual = anterior is not None and anterior.replace("\r\n", "\n") == saida.replace("\r\n", "\n")
        if not args.conferir:
            destino.write_text(saida, encoding="utf-8", newline="\n")
        (iguais if igual else mudou).append(arquivo)
        print(f"  {'igual   ' if igual else 'MUDOU   '} {nome:<22} -> {arquivo}")

    print(f"\nresumo: {len(iguais)} igual(is) | {len(mudou)} mudou/mudaram | {len(falhou)} falha(s)")
    if mudou and not args.conferir:
        print("        confira com: git diff outputs/")
    if ORFAS:
        print(f"\nsem script atual (registro historico): {', '.join(ORFAS)}")
    return 1 if falhou else 0


if __name__ == "__main__":
    sys.exit(main())
