# Projeto 1 — Do texto à decisão: sentimento e temas nas avaliações da Americanas

Projeto Integrado: Redes Sociais e Marketing · CDIA, 4º período · PUC-SP · 2026.2

## Equipe e divisão do trabalho

| # | Nome completo | Matrícula | Responsabilidade principal |
|---|---|---|---|
| 1 | Matheus Campos Amorim | 00361482 | *(preencher)* |
| 2 | *(preencher)* | | |
| 3 | *(preencher)* | | |
| 4 | *(preencher)* | | |
| 5 | *(preencher)* | | |

Todos os integrantes devem saber explicar qualquer célula na arguição individual.

## A base e a licença

**B2W-Reviews01** — 132.373 avaliações de produtos da Americanas.com (01/01 a 31/05/2018), baixada do repositório
oficial: <https://github.com/americanas-tech/b2w-reviews01> (arquivo `B2W-Reviews01.csv`, pela URL *raw*). Nenhuma
cópia (Kaggle, Hugging Face) foi usada. **A base não vai no zip**: o notebook 01 a baixa para `dados/`.

Licença **CC BY-NC-SA 4.0** — uso não comercial, com atribuição:

> REAL, L.; OSHIRO, M.; MAFRA, A. *B2W-Reviews01: an open product reviews corpus*. STIL — Symposium in Information and
> Human Language Technology, 2019.

Dados pessoais: o `reviewer_id` não é usado em nenhuma análise; gênero, estado e idade aparecem só em tabelas
agregadas (estados com ≥ 1.000 avaliações), nunca associados a uma avaliação individual.

## Estrutura

```
equipe-N-p1/
├── README.md
├── requirements.txt
├── 01_auditoria.ipynb     auditoria, as 4 perguntas, decisão do rótulo, baseline
├── 02_sentimento.ipynb    pré-processamento medido, TF-IDF × Word2Vec, 3 classificadores, vazamento, 10 erros
├── 03_temas.ipynb         NMF (artefatos, polaridade, categoria), contraprova Ward/K-Means, recomendações, limites
├── relatorio.pdf          (a produzir pela equipe)
├── apresentacao.pdf       (a produzir pela equipe)
├── figuras/               gráficos gerados pelos notebooks
├── artefatos/             base auditada, decisões e placar (gerados pelos notebooks)
├── .streamlit/config.toml tema do app
└── app/
    ├── app.py             app Streamlit (opcional, +1,0)
    ├── preproc.py         pré-processamento (gerado pela célula 02.02; o mesmo do treino)
    ├── requirements.txt
    └── modelos/           sentimento.joblib (notebook 02) e temas.joblib (notebook 03)
```

## Como executar

Os notebooks rodam **na ordem** 01 → 02 → 03, cada um com *Restart & Run All*. O 01 grava
`artefatos/base_auditada.parquet`, que os outros leem; o 02 grava `app/preproc.py`, que o 03 e o app usam.

**Local** (Python ≥ 3.10):

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_auditoria.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_sentimento.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_temas.ipynb
streamlit run app/app.py
```

**Google Colab:** suba a pasta inteira (ou clone-a no Drive), abra cada notebook com o diretório de trabalho na raiz da
pasta (`%cd /content/drive/MyDrive/equipe-N-p1`) e rode na ordem. As células iniciais instalam `gensim`, `nltk` e
`pyarrow` se faltarem.

Tempo aproximado numa máquina de 1 núcleo: 01 ≈ 1 min · 02 ≈ 5 min · 03 ≈ 4 min. Os notebooks usam a **base inteira**,
sem amostra (exceto onde declarado: 40.000 avaliações na reprodução da tabela do professor e 3.000 na contraprova
hierárquica).

**Reprodutibilidade:** semente 42 em todas as divisões, no NMF, no K-Means e no Word2Vec (`seed=42, workers=1`). O
vetorizador e o Word2Vec são ajustados só no treino.

## Resultados principais (cada número com a célula de origem)

| item | resultado | célula |
|---|---|---|
| rótulo | nota binária: 1-2★ negativo, 4-5★ positivo, 3★ fora (88,5% das 3★ recomendam) | 01.10 |
| conjunto do modelo | 110.874 avaliações (sem vazias, sem duplicatas) | 01.23 |
| baseline (classe majoritária) | **0,706** | 01.23 |
| melhor modelo (teste) | TF-IDF (1,2) + regressão logística: acurácia **0,953**, macro-F1 **0,943** | 02.17 |
| Word2Vec médio (teste) | acurácia 0,935, macro-F1 0,922 | 02.17 |
| vazamento pelo título | com título 0,970 (−36% de erros); o número honesto é o sem título | 02.23 |
| classe neutra | recall 0,228 (0,565 com pesos, a −14 pontos no positivo) | 02.25 |
| temas negativos | 8 temas; família Entrega = 24,2% das negativas | 03.09, 03.19 |
| Móveis | 52,4% de negativas; "peças faltando e montagem" = 32,5% delas | 03.20, 03.12 |

## App Streamlit

`streamlit run app/app.py` — cole uma avaliação e veja a classe prevista, a probabilidade de cada classe, o texto com
cada palavra colorida pelo peso que teve na decisão e o tema mais próximo (NMF da classe prevista). O app só **carrega**
os modelos de `app/modelos/`; não retreina.

## Uso de IA generativa (declaração exigida pelo enunciado)

*(A equipe deve revisar e completar este parágrafo, e repeti-lo no relatório.)* Usamos o **Claude (Anthropic)** como
ferramenta de apoio para: estruturar e escrever o código dos três notebooks e do app, executar os notebooks, e redigir
um rascunho das análises a partir das saídas. Todas as decisões foram revisadas pela equipe, que deve ser capaz de
explicar cada célula.

## Referências

* REAL, L.; OSHIRO, M.; MAFRA, A. B2W-Reviews01: an open product reviews corpus. STIL, 2019.
* Documentação oficial do scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `MultinomialNB`, `NMF`, `TruncatedSVD`,
  `KMeans`, métricas), do gensim (`Word2Vec`), do SciPy (`linkage`, `fcluster`) e do pandas.
* NLTK: lista de stop words do português e stemmer RSLP.
* Materiais da disciplina: Lab 05 (*Do texto ao vetor*) e Prática 03 (*agrupamento hierárquico*).
