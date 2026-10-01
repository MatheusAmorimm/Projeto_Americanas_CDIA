# ⟦célula 02.02⟧
"""Pré-processamento do Projeto 1 (B2W-Reviews01). Gerado pelo notebook 02, célula 02.02."""
import re, unicodedata

URL = re.compile(r"https?://\S+|www\.\S+")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0000FE0F]")
ESTICADA = re.compile(r"([^\W\d_])\1{2,}")          # letra repetida 3+ vezes: muuuito -> muito
SO_LETRAS = re.compile(r"[^a-zà-ÿ\s]")

# NLTK 'portuguese' (2024), sem acentos -- 200 formas
STOPWORDS_NLTK = set("""a ao aos aquela aquelas aquele aqueles aquilo as ate com como da das de dela delas dele deles
depois do dos e ela elas ele eles em entre era eram eramos essa essas esse esses esta estamos estao estar estas estava
estavam estavamos este esteja estejam estejamos estes esteve estive estivemos estiver estivera estiveram estiveramos
estiverem estivermos estivesse estivessem estivessemos estou eu foi fomos for fora foram foramos forem formos fosse
fossem fossemos fui ha haja hajam hajamos hao havemos haver hei houve houvemos houver houvera houveram houveramos
houverao houverei houverem houveremos houveria houveriam houveriamos houvermos houvesse houvessem houvessemos isso
isto ja lhe lhes mais mas me mesmo meu meus minha minhas muito na nao nas nem no nos nossa nossas nosso nossos num
numa o os ou para pela pelas pelo pelos por qual quando que quem sao se seja sejam sejamos sem ser sera serao serei
seremos seria seriam seriamos seu seus so somos sou sua suas tambem te tem temos tenha tenham tenhamos tenho tera
terao terei teremos teria teriam teriamos teu teus teve tinha tinham tinhamos tive tivemos tiver tivera tiveram
tiveramos tiverem tivermos tivesse tivessem tivessemos tu tua tuas um uma voce voces vos""".split())

# negação, intensidade e contraste: carregam sentimento e NÃO podem sair
PROTEGIDAS = {"nao", "nem", "nunca", "sem", "mas", "muito", "mais", "ja", "so", "ate", "mesmo"}
STOPWORDS = STOPWORDS_NLTK - PROTEGIDAS
MARCA_CAIXA_ALTA = "_caixa_alta_"


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def fracao_maiuscula(s):
    letras = [c for c in s if c.isalpha()]
    return sum(c.isupper() for c in letras) / len(letras) if letras else 0.0


def normalizar(texto, acentos=False, esticada=True, caixa_alta=True):
    """Texto cru -> tokens normalizados, separados por espaço (antes de stop words e stemming)."""
    texto = "" if texto is None else str(texto)
    grita = caixa_alta and len(texto) >= 10 and fracao_maiuscula(texto) >= 0.8
    s = texto.lower()
    s = URL.sub(" ", s)
    s = EMOJI.sub(" ", s)
    if not acentos:
        s = sem_acento(s)
    s = SO_LETRAS.sub(" ", s)
    if esticada:
        s = ESTICADA.sub(r"\1", s)
    tokens = s.split()
    if grita:
        tokens.append(MARCA_CAIXA_ALTA)
    return " ".join(tokens)


def tokenizar(normalizado, stop=STOPWORDS, stemmer=None):
    tokens = [t for t in normalizado.split() if sem_acento(t) not in stop] if stop else normalizado.split()
    if stemmer is not None:
        tokens = [t if t == MARCA_CAIXA_ALTA else stemmer(t) for t in tokens]
    return tokens


def preparar(texto, acentos=False, esticada=True, caixa_alta=True, remover_stop=True):
    """O pipeline completo, na configuração escolhida -- usado pelo app."""
    return " ".join(tokenizar(normalizar(texto, acentos, esticada, caixa_alta),
                              stop=STOPWORDS if remover_stop else None))
