"""App do Projeto 1 — classifica o sentimento de uma avaliação e aponta o tema mais próximo.

Rodar (a partir da pasta da entrega, depois de executar os notebooks 02 e 03):
    streamlit run app/app.py

Os modelos são carregados de app/modelos/*.joblib (gerados pelos notebooks); nada é retreinado aqui.
"""
import html
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
import preproc  # noqa: E402  (mesmo módulo usado no treino, gerado pela célula 02.02)

POSITIVO, NEGATIVO, TINTA, NEUTRO = "#2E5E8C", "#B4532A", "#1F2A36", "#8A94A0"

st.set_page_config(page_title="Sentimento das avaliações · Americanas 2018", page_icon="🛒", layout="centered")


@st.cache_resource
def carregar():
    sent = joblib.load(PASTA / "modelos" / "sentimento.joblib")
    temas = joblib.load(PASTA / "modelos" / "temas.joblib")
    return sent, temas


try:
    SENT, TEMAS = carregar()
except FileNotFoundError:
    st.error("Os modelos não foram encontrados em app/modelos/. Execute os notebooks 02_sentimento e 03_temas "
             "(eles gravam sentimento.joblib e temas.joblib) e abra o app de novo.")
    st.stop()

VET, MODELO, CONFIG = SENT["vetorizador"], SENT["modelo"], SENT["config"]
CLASSES = list(MODELO.classes_)
COEF = MODELO.coef_[0]                       # > 0 puxa para a classe CLASSES[1] ("positivo")
NOMES_COLUNAS = VET.get_feature_names_out()
STOP_TEMAS = set(TEMAS["stop_temas"])


def classificar(texto):
    doc = preproc.preparar(texto, **CONFIG)
    x = VET.transform([doc])
    proba = dict(zip(CLASSES, MODELO.predict_proba(x)[0]))
    contrib = {NOMES_COLUNAS[j]: COEF[j] * v for j, v in zip(x.indices, x.data)}
    return doc, proba, contrib


def tema_mais_proximo(texto, classe):
    m = TEMAS[classe]
    norm = preproc.normalizar(texto, esticada=True, caixa_alta=False)
    doc = " ".join(w for w in norm.split() if w not in STOP_TEMAS)
    w = m["nmf"].transform(m["vetorizador"].transform([doc]))[0]
    if w.sum() == 0:
        return None
    ordem = np.argsort(-w)
    return [(m["nomes"][i], m["familia"][i], w[i] / w.sum()) for i in ordem[:3] if w[i] > 0]


def texto_destacado(doc, contrib):
    """Cada token do texto processado, com fundo proporcional ao peso que teve (unigrama + bigramas em que entra)."""
    tokens = doc.split()
    peso = np.zeros(len(tokens))
    for k, t in enumerate(tokens):
        peso[k] += contrib.get(t, 0.0)
        if k + 1 < len(tokens):
            b = contrib.get(f"{t} {tokens[k + 1]}", 0.0) / 2   # bigrama: metade para cada palavra
            peso[k] += b
            peso[k + 1] += b
    escala = max(np.abs(peso).max(), 1e-9)
    partes = []
    for t, p in zip(tokens, peso):
        alfa = min(abs(p) / escala, 1.0) * 0.55
        cor = POSITIVO if p > 0 else NEGATIVO
        r, g, b = (int(cor[i:i + 2], 16) for i in (1, 3, 5))
        fundo = f"rgba({r},{g},{b},{alfa:.2f})" if abs(p) > 1e-6 else "transparent"
        titulo = f"peso {p:+.2f}" if abs(p) > 1e-6 else "sem peso no modelo"
        partes.append(f'<span title="{titulo}" style="background:{fundo};padding:2px 3px;border-radius:3px">{html.escape(t)}</span>')
    return " ".join(partes)


# ------------------------------------------------------------------ interface
st.markdown(f"""
<style>
  .bloco-texto {{ font-size: 1.15rem; line-height: 2.1; color: {TINTA}; }}
  .veredito {{ font-size: 2.1rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }}
  .nota {{ color: {NEUTRO}; font-size: 0.88rem; }}
</style>
""", unsafe_allow_html=True)

st.title("O que essa avaliação diz?")
st.write("Cole o texto de uma avaliação de produto. O modelo diz se ela é positiva ou negativa, mostra quais palavras "
         "pesaram na decisão e de que assunto ela trata.")

EXEMPLOS = {
    "Móvel incompleto": "Chegou rápido, mas veio faltando peças e os parafusos. Estou há duas semanas sem conseguir montar o guarda-roupa.",
    "Elogio à entrega": "Produto chegou antes do prazo, bem embalado e funcionando perfeitamente. Recomendo!",
    "Ainda não chegou": "Como posso avaliar se ainda não recebi o produto? O prazo já passou e ninguém responde.",
    "Ironia": "Maravilha, o celular trava toda hora e a bateria dura duas horas. Excelente compra, parabéns.",
}
if "texto" not in st.session_state:
    st.session_state.texto = ""
colunas = st.columns(len(EXEMPLOS))
for col, (rotulo, exemplo) in zip(colunas, EXEMPLOS.items()):
    if col.button(rotulo, use_container_width=True):
        st.session_state.texto = exemplo

texto = st.text_area("Texto da avaliação", key="texto", height=120,
                     placeholder="Ex.: O produto veio com defeito e a troca está demorando…")

if not texto.strip():
    st.info("Escreva ou cole uma avaliação acima, ou escolha um dos exemplos, para ver a análise.")
else:
    doc, proba, contrib = classificar(texto)
    if not contrib:
        st.warning("Nenhuma palavra deste texto está no vocabulário do modelo, então a previsão abaixo é só a "
                   "tendência geral da base. Tente um texto mais longo, em português.")
    classe = max(proba, key=proba.get)
    cor = POSITIVO if classe == "positivo" else NEGATIVO

    st.markdown(f'<p class="veredito" style="color:{cor}">{classe.capitalize()}</p>'
                f'<p class="nota">{proba[classe]:.0%} de probabilidade segundo o modelo</p>', unsafe_allow_html=True)
    for c in ["negativo", "positivo"]:
        st.progress(float(proba[c]), text=f"{c}: {proba[c]:.1%}")

    st.subheader("As palavras que pesaram")
    st.markdown(f'<div class="bloco-texto">{texto_destacado(doc, contrib)}</div>', unsafe_allow_html=True)
    st.caption("Este é o texto como o modelo o vê (minúsculas, sem acento, sem stop words). Azul puxa para positivo, "
               "ferrugem para negativo; quanto mais forte a cor, maior o peso. Passe o mouse sobre uma palavra para ver "
               "o valor.")

    ordenado = sorted(contrib.items(), key=lambda kv: kv[1])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Puxaram para negativo**")
        neg = [(t, v) for t, v in ordenado if v < 0][:6]
        st.markdown("\n".join(f"- `{t}` ({v:+.2f})" for t, v in neg) or "nenhuma")
    with c2:
        st.markdown("**Puxaram para positivo**")
        pos = [(t, v) for t, v in reversed(ordenado) if v > 0][:6]
        st.markdown("\n".join(f"- `{t}` ({v:+.2f})" for t, v in pos) or "nenhuma")

    st.subheader("Tema mais próximo")
    temas = tema_mais_proximo(texto, classe)
    if temas is None:
        st.write("Nenhum tema reconhecido: o texto não usa as palavras que definem os temas das avaliações "
                 f"{'negativas' if classe == 'negativo' else 'positivas'}.")
    else:
        nome, familia, parte = temas[0]
        st.markdown(f"**{nome}** — família *{familia}*, {parte:.0%} do peso dos temas neste texto.")
        if len(temas) > 1:
            st.caption("Também presentes: " + "; ".join(f"{n} ({p:.0%})" for n, _, p in temas[1:]))
        st.caption(f"Temas aprendidos (NMF, k = 8) nas avaliações {'negativas' if classe == 'negativo' else 'positivas'} "
                   "da base; o modelo usado é o da classe prevista.")

with st.sidebar:
    st.header("Sobre o modelo")
    st.write(f"TF-IDF (unigramas e bigramas, `min_df={SENT['min_df']}`) e regressão logística, treinados em "
             f"{SENT['treinado_em']}.")
    st.metric("Acurácia no teste", f"{SENT['acuracia_teste']:.3f}",
              help="Medida em 22.175 avaliações nunca vistas no treino.")
    st.metric("Chute na classe majoritária", f"{SENT['baseline_teste']:.3f}",
              help="Acerto de quem responde sempre 'positivo'. Toda acurácia deve ser lida ao lado deste número.")
    st.metric("Macro-F1 no teste", f"{SENT['macro_f1_teste']:.3f}")
    st.markdown("**Limites.** Avaliações da Americanas.com de jan. a mai. de 2018; só 1-2★ (negativo) e 4-5★ (positivo), "
                "a nota 3 ficou de fora. O modelo conta palavras e não lê a frase: ironia e negação distante enganam "
                "(teste o exemplo *Ironia*).")
    st.caption("Base: B2W-Reviews01 — Real, L.; Oshiro, M.; Mafra, A. (STIL 2019), licença CC BY-NC-SA 4.0. "
               "Projeto Integrado: Redes Sociais e Marketing · CDIA · PUC-SP · 2026.2.")
