# Dados

Este diretório só recebe `solicitacoes_sinteticas.csv`, gerado por `src/gerar_dados.py`
(semente fixa; o sha256 sai em `outputs/00-dados-sinteticos.txt`). O CSV não é versionado
porque é regenerado de forma determinística a cada `rodar_tudo.py`.

**É dado sintético.** O schema (15 colunas e o domínio de cada uma) está em
`src/prepare.py`; as distribuições e os coeficientes do gerador foram escritos à mão. O
efeito na região Norte (risco que depende de um fator ausente do arquivo) foi **plantado**.
Nenhuma linha vem de dado real.
