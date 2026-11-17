
import streamlit as st
import torch
import torch.nn.functional as F
import os
import time

from modelo import MiGeminiDesdeCero
from tokenizador import TokenizadorLocal

st.set_page_config(
    page_title="Migo IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- DISEÑO ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background:
      radial-gradient(ellipse at 50% 0%, #142b50 0%, transparent 45%),
      #080d18;
    color: #edf4ff;
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stSidebar"] {
    background: #0d1628;
    border-right: 1px solid #203653;
}
.hero {
    padding: 35px 10px 25px;
    text-align: center;
}
.logo {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 4px;
    color: #65baff;
    margin-bottom: 12px;
}
.hero h1 {
    font-size: clamp(42px, 7vw, 68px);
    font-weight: 800;
    letter-spacing: -3px;
    background: linear-gradient(90deg,#ffffff,#64b5ff,#a78bfa);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0;
}
.hero p {
    color: #9aacc7;
    font-size: 16px;
    margin-top: 12px;
}
.status {
    display: inline-block;
    padding: 8px 15px;
    margin-top: 15px;
    border: 1px solid #245b50;
    border-radius: 30px;
    background: #102b2a;
    color: #75e0bd;
    font-size: 12px;
}
div[data-testid="stChatMessage"] {
    background: #111d31;
    border: 1px solid #233956;
    border-radius: 16px;
    padding: 15px;
    margin-bottom: 12px;
}
[data-testid="stChatInput"] {
    border: 1px solid #376da8;
    border-radius: 16px;
    background: #101b2d;
}
[data-testid="stChatInput"] textarea {
    color: #ffffff;
}
.stButton button {
    border-radius: 10px;
    border: 1px solid #315b88;
    background: #142844;
    color: #eaf4ff;
}
.stButton button:hover {
    border-color: #62b5ff;
    color: #ffffff;
}
div[data-testid="stMetric"] {
    background: #111d31;
    border: 1px solid #233956;
    border-radius: 12px;
    padding: 12px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- MODELO ----------
RUTA_TEXTO = "datos_entrenamiento.txt"
RUTA_PESOS = "mi_gemini_pesos.pth"
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def cargar_cerebro_ia():
    if not os.path.exists(RUTA_TEXTO):
        return None, None, None

    tokenizador = TokenizadorLocal(RUTA_TEXTO)
    modelo = MiGeminiDesdeCero(
        tamano_vocabulario=tokenizador.tamano_vocabulario
    ).to(dispositivo)

    checkpoint_info = None

    if os.path.exists(RUTA_PESOS):
        checkpoint_info = torch.load(
            RUTA_PESOS,
            map_location=dispositivo
        )

        if isinstance(checkpoint_info, dict) and "modelo" in checkpoint_info:
            modelo.load_state_dict(checkpoint_info["modelo"])
        else:
            modelo.load_state_dict(checkpoint_info)

        modelo.eval()

    return modelo, tokenizador, checkpoint_info

with st.spinner("🧠 Preparando Migo IA..."):
    modelo, tokenizador, checkpoint_info = cargar_cerebro_ia()

# ---------- BARRA LATERAL ----------
with st.sidebar:
    st.markdown("## ⚙️ Ajustes")

    temperatura = st.slider(
        "Creatividad",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1
    )

    max_tokens = st.slider(
        "Longitud de respuesta",
        min_value=20,
        max_value=300,
        value=100,
        step=10
    )

    st.divider()
    st.markdown("## 🧠 Estado de Migo")

    if modelo is None or tokenizador is None:
        st.error("No se han encontrado los datos de entrenamiento.")
    elif not os.path.exists(RUTA_PESOS):
        st.warning("El modelo todavía no tiene pesos entrenados.")
    else:
        st.success("Modelo cargado y listo")

    if isinstance(checkpoint_info, dict) and "paso" in checkpoint_info:
        st.metric("Pasos de entrenamiento", checkpoint_info["paso"])

    st.caption(f"Dispositivo: {dispositivo.upper()}")

    if tokenizador is not None:
        st.caption(f"Vocabulario: {tokenizador.tamano_vocabulario} caracteres")

    st.divider()

    if st.button("🗑️ Borrar conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

    st.caption("Migo IA · Proyecto independiente")

# ---------- CABECERA ----------
st.markdown("""
<div class="hero">
    <div class="logo">INTELIGENCIA ARTIFICIAL INDEPENDIENTE</div>
    <h1>MIGO IA</h1>
    <p>Tu asistente experimental, construido desde cero.</p>
    <div class="status">● Sistema conectado</div>
</div>
""", unsafe_allow_html=True)

# ---------- CONVERSACIÓN ----------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for mensaje in st.session_state.mensajes:
    avatar = "🧑" if mensaje["rol"] == "user" else "🤖"
    with st.chat_message(mensaje["rol"], avatar=avatar):
        st.markdown(mensaje["texto"])

entrada_usuario = st.chat_input("Escribe un mensaje a Migo IA...")

if entrada_usuario:
    st.session_state.mensajes.append({
        "rol": "user",
        "texto": entrada_usuario
    })

    with st.chat_message("user", avatar="🧑"):
        st.markdown(entrada_usuario)

    with st.chat_message("assistant", avatar="🤖"):
        contenedor = st.empty()

        if modelo is None or tokenizador is None:
            respuesta = (
                "No encuentro `datos_entrenamiento.txt`. "
                "Comprueba que el archivo esté en el repositorio."
            )
            contenedor.markdown(respuesta)

        elif not os.path.exists(RUTA_PESOS):
            respuesta = (
                "Todavía no tengo pesos entrenados. "
                "Entréname primero y vuelve a intentarlo."
            )
            contenedor.markdown(respuesta)

        else:
            frase = f"<|usuario|> {entrada_usuario}\n<|asistente|>"
            contexto = tokenizador.codificar(frase)
            x = torch.tensor(
                [contexto],
                dtype=torch.long,
                device=dispositivo
            )

            respuesta = ""
            tokens_generados = []

            with torch.no_grad():
                for _ in range(max_tokens):
                    x_cond = x[:, -64:]
                    logits = modelo(x_cond)
                    logits = logits[:, -1, :] / temperatura

                    probs = F.softmax(logits, dim=-1)
                    siguiente_token = torch.multinomial(probs, num_samples=1)
                    id_token = siguiente_token.item()

                    palabra = tokenizador.int_a_token.get(id_token, "")

                    if palabra in ("<|usuario|>", "<|asistente|>"):
                        break

                    tokens_generados.append(id_token)
                    respuesta = tokenizador.decodificar(tokens_generados)

                    contenedor.markdown(respuesta + " ▌")
                    x = torch.cat((x, siguiente_token), dim=1)
                    time.sleep(0.01)

            if not respuesta.strip():
                respuesta = "Necesito más entrenamiento para responder mejor."

            contenedor.markdown(respuesta)

    st.session_state.mensajes.append({
        "rol": "assistant",
        "texto": respuesta
    })
