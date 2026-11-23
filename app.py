import hmac
import json
import os
import threading

import streamlit as st

import nucleo as n

# ==========================================================
# AJUSTES
# ==========================================================
NOMBRE = "Migo IA"
VERSION = "1.1"
MODELO_REPO = "bartowski/Qwen_Qwen3-0.6B-GGUF"   # Más listo pero necesita más memoria: "bartowski/Qwen_Qwen3-1.7B-GGUF"
MODELO_ARCHIVO = "*Q4_K_M.gguf"                  # Patrón del archivo dentro del repositorio
MODELO_NOMBRE = "Qwen3-0.6B"
N_CTX = 1536                  # Memoria de la conversación (en tokens). Más = más RAM
MAX_TOKENS = 384              # Largo máximo de cada respuesta
PRESUPUESTO_HISTORIAL = 3000  # Caracteres de conversación que se envían al modelo

SISTEMA = (
    f"Eres {NOMBRE} {VERSION}, un asistente de IA amable que conversa en español de forma natural, "
    "clara y breve. Si no sabes algo, dilo con honestidad en vez de inventar. "
    f"Si te preguntan por tu tecnología, explica con sinceridad que {NOMBRE} funciona sobre el "
    f"modelo abierto {MODELO_NOMBRE}. /no_think"
)


def secreto(nombre, defecto=""):
    try:
        return st.secrets[nombre]
    except Exception:
        return os.environ.get(nombre, defecto)


def puerta():
    """Si pones CLAVE_APP en los secretos de Streamlit, la app pide contraseña."""
    clave = secreto("CLAVE_APP")
    if not clave or st.session_state.get("autorizado"):
        return
    st.title(f"🧠 {NOMBRE}")
    intento = st.text_input("Contraseña", type="password")
    if intento:
        if hmac.compare_digest(intento.encode(), str(clave).encode()):
            st.session_state.autorizado = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()


@st.cache_resource(show_spinner="Cargando el cerebro de Migo IA… (la primera vez tarda un par de minutos)", max_entries=1)
def cargar_modelo():
    from llama_cpp import Llama
    llm = Llama.from_pretrained(
        repo_id=MODELO_REPO, filename=MODELO_ARCHIVO, n_ctx=N_CTX,
        n_threads=min(2, os.cpu_count() or 2), verbose=False,
    )
    return llm, threading.Lock()   # El candado evita que dos visitantes usen el modelo a la vez


def generar(llm, lock, historial_api, sistema):
    mensajes = [{"role": "system", "content": sistema}] + historial_api
    with lock:
        flujo = llm.create_chat_completion(
            messages=mensajes, max_tokens=MAX_TOKENS, temperature=0.7,
            top_p=0.8, top_k=20, repeat_penalty=1.05, stream=True,
        )
        crudo, mostrado = "", ""
        for trozo in flujo:
            delta = trozo["choices"][0].get("delta", {}).get("content")
            if not delta:
                continue
            crudo += delta
            visible = n.texto_visible(crudo)
            if len(visible) > len(mostrado):
                yield visible[len(mostrado):]
                mostrado = visible


# ==========================================================
# INTERFAZ
# ==========================================================
st.set_page_config(page_title=f"{NOMBRE} {VERSION}", page_icon="🧠")
puerta()
st.title(f"🧠 {NOMBRE} {VERSION}")
st.caption(f"Modelo propio dentro de esta app: {MODELO_NOMBRE}. Escribe «Aprende: …» para enseñarme algo.")

if "hechos" not in st.session_state:
    st.session_state.hechos = n.cargar_hechos()
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
hechos = st.session_state.hechos

with st.sidebar:
    st.subheader("🧠 Memoria")
    st.markdown(f"**Cosas aprendidas:** `{len(hechos)}`")
    if hechos:
        with st.expander("Ver lo que sé"):
            for h in hechos:
                st.markdown(f"- {h}")
    st.download_button("💾 Descargar memoria", json.dumps(hechos, ensure_ascii=False, indent=2),
                       file_name="migo_memoria.json", mime="application/json", use_container_width=True)
    subido = st.file_uploader("Cargar memoria (migo_memoria.json)", type="json")
    if subido is not None:
        huella = (subido.name, subido.size)
        if st.session_state.get("ultima_carga") != huella:
            st.session_state.ultima_carga = huella
            nuevos = n.importar(hechos, subido.getvalue().decode("utf-8"))
            st.success(f"{nuevos} recuerdos nuevos cargados")
    st.divider()
    if st.button("🗑️ Borrar conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

entrada = st.chat_input("Escribe tu mensaje…")
if entrada:
    with st.chat_message("user"):
        st.markdown(entrada)

    respuesta_local = n.comando_memoria(entrada, hechos)
    if respuesta_local:
        with st.chat_message("assistant"):
            st.markdown(respuesta_local)
        st.session_state.mensajes += [
            {"role": "user", "content": entrada, "local": True},
            {"role": "assistant", "content": respuesta_local, "local": True},
        ]
    else:
        try:
            llm, lock = cargar_modelo()
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")
            st.stop()

        historial = [{"role": m["role"], "content": m["content"]}
                     for m in st.session_state.mensajes if not m.get("local")]
        historial.append({"role": "user", "content": entrada})
        historial = n.recortar_historial(historial, PRESUPUESTO_HISTORIAL)
        sistema = n.construir_sistema(SISTEMA, hechos)

        with st.chat_message("assistant"):
            hueco = st.empty()
            hueco.markdown("💭 *Pensando…*")
            acumulado = ""
            try:
                for delta in generar(llm, lock, historial, sistema):
                    acumulado += delta
                    hueco.markdown(acumulado + "▌")
            except Exception as e:
                hueco.error(f"Error al generar la respuesta: {e}")
                st.stop()
            acumulado = acumulado.strip() or "_No pude generar una respuesta, inténtalo de nuevo._"
            hueco.markdown(acumulado)

        st.session_state.mensajes += [
            {"role": "user", "content": entrada},
            {"role": "assistant", "content": acumulado},
        ]
