"""Veste o relatório para leitura no navegador (GitHub Pages) e para virar PDF.

Le `outputs/09-relatorio-de-confianca.md` e devolve HTML no stdout, que o `rodar_tudo.py`
grava como `outputs/09-relatorio-de-confianca.html`. A mesma pagina, com as figuras, vai
para `docs/index.html`, que e' o que o GitHub Pages publica.

O PDF sai do HTML por qualquer Chrome/Chromium headless (`--print-to-pdf`), que respeita a
folha de estilo de impressao.
"""
from __future__ import annotations

from pathlib import Path

import markdown

RAIZ = Path(__file__).resolve().parents[1]
FONTE = RAIZ / "outputs" / "09-relatorio-de-confianca.md"
PAGES = RAIZ / "docs" / "index.html"
FIGURAS = [
    ("custo_limiar.png", "Custo de operar em cada limiar (cenário 10:1). O limiar é escolhido "
     "na calibração; a curva é medida no teste."),
    ("recall_por_regiao.png", "Recall por região, com intervalo de 95% (Wilson). O efeito no "
     "Norte foi plantado no gerador sintético."),
    ("calibracao.png", "Diagrama de confiabilidade no teste: probabilidade prevista contra "
     "taxa observada, em oito faixas de mesma população."),
]

ESTILO = """
  :root {
    --papel: #FFFFFF; --papel-2: #F5F8F5; --tinta: #14261F; --tinta-suave: #47594F;
    --linha: #CBD8CC; --fala: #1F5D45; --ambar: #8A6008; --tijolo: #9E3925;
    --display: "Bricolage Grotesque", "Helvetica Neue", Arial, sans-serif;
    --corpo: "Literata", Georgia, "Times New Roman", serif;
    --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --papel: #0F1613; --papel-2: #17211C; --tinta: #E4EBE4; --tinta-suave: #9DB0A4;
      --linha: #2B3830; --fala: #6FC69E; --ambar: #E2AE4C; --tijolo: #E68C74;
    }
  }
  :root[data-theme="dark"] {
    --papel: #0F1613; --papel-2: #17211C; --tinta: #E4EBE4; --tinta-suave: #9DB0A4;
    --linha: #2B3830; --fala: #6FC69E; --ambar: #E2AE4C; --tijolo: #E68C74;
  }
  * { box-sizing: border-box; }
  body { background: var(--papel); color: var(--tinta); font-family: var(--corpo);
         font-size: 15px; line-height: 1.55; margin: 0 auto; padding: 48px 28px 72px;
         max-width: 840px; }
  h1 { font-family: var(--display); font-size: 38px; font-weight: 700; letter-spacing: -.02em;
       line-height: 1.05; margin: 0 0 6px; text-wrap: balance; }
  h1 + p { color: var(--tinta-suave); }
  h2 { font-family: var(--display); font-size: 21px; font-weight: 700; margin: 34px 0 12px;
       padding-top: 14px; border-top: 2px solid var(--tinta); letter-spacing: -.01em; }
  h3 { font-family: var(--display); font-size: 16px; font-weight: 700; margin: 22px 0 9px; }
  p, li { margin: 0 0 11px; }
  ul, ol { padding-left: 22px; margin: 0 0 12px; }
  strong { font-weight: 600; }
  hr { border: 0; height: 0; margin: 0; }
  table { border-collapse: collapse; width: 100%; margin: 6px 0 18px; font-size: 13.5px;
          font-variant-numeric: tabular-nums; }
  th { text-align: left; font-family: var(--mono); font-size: 10.5px; letter-spacing: .09em;
       text-transform: uppercase; color: var(--tinta-suave); font-weight: 600;
       border-bottom: 1.5px solid var(--tinta); padding: 7px 10px 7px 0; }
  td { padding: 7px 10px 7px 0; border-bottom: 1px solid var(--linha); vertical-align: top; }
  tr:last-child td { border-bottom: 0; }
  blockquote { margin: 14px 0; padding: 13px 18px; background: var(--papel-2);
               border-left: 3px solid var(--ambar); border-radius: 0 3px 3px 0;
               font-size: 14.5px; }
  blockquote p:last-child { margin-bottom: 0; }
  code { font-family: var(--mono); font-size: 12.5px; background: var(--papel-2);
         padding: 1px 4px; border-radius: 2px; }
  @media print {
    @page { size: A4; margin: 16mm 14mm; }
    body { font-size: 9.5pt; line-height: 1.4; padding: 0; max-width: none;
           background: #fff; color: #000; }
    h1 { font-size: 22pt; } h2 { font-size: 13pt; margin-top: 20px; padding-top: 9px; }
    h3 { font-size: 11pt; }
    h2, h3 { break-after: avoid; }
    table, blockquote { break-inside: avoid; }
    table { font-size: 8.5pt; } blockquote { font-size: 9pt; background: #f4f6f4; }
    code { background: transparent; }
  }
"""

CABECA = """<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700&family=IBM+Plex+Mono:wght@400;600&family=Literata:opsz,wght@7..72,400;7..72,600&display=swap">"""


def main() -> None:
    corpo = markdown.markdown(
        FONTE.read_text(encoding="utf-8"),
        extensions=["tables", "sane_lists"],
    )
    pagina = (f"<!doctype html>\n<html lang=\"pt-BR\">\n<meta charset=\"utf-8\">\n"
              f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
              f"<title>Relatório de Confiança: modelo de crédito (dado sintético)</title>\n"
              f"{CABECA}\n<style>{ESTILO}</style>\n")
    print(pagina + corpo)

    figuras = "".join(
        f'<figure><img src="figuras/{arq}" alt="{leg}" style="max-width:100%">'
        f"<figcaption>{leg}</figcaption></figure>\n"
        for arq, leg in FIGURAS if (RAIZ / "docs" / "figuras" / arq).exists())
    PAGES.parent.mkdir(exist_ok=True)
    PAGES.write_text(pagina + corpo + "\n<h2>Figuras</h2>\n" + figuras, encoding="utf-8")


if __name__ == "__main__":
    main()
