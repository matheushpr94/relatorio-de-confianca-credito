NUMEROS CITADOS NO README (dado sintetico)

[dado] linhas 18,000 | prevalencia 7.60% | teste 3,600 com 273 inadimplentes
[dado] taxa de calote Centro-Oeste  8.70%
[dado] taxa de calote Nordeste      7.44%
[dado] taxa de calote Norte         7.38%
[dado] taxa de calote Sudeste       7.40%
[dado] taxa de calote Sul           8.02%

[modelos] AUPRC na calibracao (piso = prevalencia 0.0761)
   regra do score        0.2046  (2.7x o piso)
   logistica             0.2870  (3.8x o piso)   no proprio treino 0.2491
   floresta              0.2421  (3.2x o piso)   no proprio treino 1.0000
   boosting              0.2419  (3.2x o piso)   no proprio treino 0.5624
   ganho da logistica sobre a regra: +0.0825

[teste] AUPRC 0.2387 (piso 0.0758) | AUC 0.7224 | ECE 0.0089 | Brier 0.06476 (piso 0.07008)
[limiar] razao 10:1 -> t* 0.0660 (escolhido na calibracao)
[limiar] em t*: VP 184 FN 89 FP 1168 VN 2159 | recall 0.674 | precisao 0.136 | negados 1,352 (37.6%)
[limiar] em 0,5: recall 0.037 (10 de 273)
[limiar] custo em t* 2,058 | em 0,5 2,632 | razao 1.28x

[subgrupos] recall por regiao no limiar de referencia (IC 95% Wilson)
   Centro-Oeste   0.833 [0.641 ; 0.933]  positivos 24  
   Nordeste       0.684 [0.575 ; 0.776]  positivos 79  
   Norte          0.312 [0.180 ; 0.486]  positivos 32  PIOR
   Sudeste        0.762 [0.672 ; 0.833]  positivos 105  
   Sul            0.606 [0.437 ; 0.753]  positivos 33  
   (geral)        0.674 [0.616 ; 0.727]
[sensibilidade] Norte pior com IC fora do geral em 6 de 8 cenarios de custo

[abstencao] cobertura 90%: 360 a revisao | precisao 0.136 -> 0.143 | recall 0.674 -> 0.685
