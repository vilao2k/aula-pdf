import os
import io
import uuid
import threading
import tempfile
from functools import wraps

from flask import (
    Flask, request, redirect, url_for, session,
    render_template, send_file, jsonify, flash
)
from groq import Groq
from pydub import AudioSegment
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT

# ---------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "troque-esta-chave-em-producao")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300MB

APP_PASSWORD = os.environ.get("APP_PASSWORD", "mude-esta-senha")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Jobs em memória: job_id -> dict(status, error, pdf_bytes, filename)
JOBS = {}

CHUNK_MS = 10 * 60 * 1000  # 10 minutos por pedaço, para não estourar limites da API


# ---------------------------------------------------------------
# Autenticação simples (senha única)
# ---------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logado"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        senha = request.form.get("senha", "")
        if senha == APP_PASSWORD:
            session["logado"] = True
            return redirect(url_for("upload"))
        flash("Senha incorreta.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------
# Página de upload
# ---------------------------------------------------------------
@app.route("/", methods=["GET"])
@login_required
def upload():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
@login_required
def start_job():
    if client is None:
        return "Servidor sem GROQ_API_KEY configurada.", 500

    audio_file = request.files.get("audio")
    titulo = request.form.get("titulo", "Aula")
    if not audio_file or audio_file.filename == "":
        flash("Selecione um arquivo de áudio.")
        return redirect(url_for("upload"))

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "processing", "error": None, "pdf_bytes": None, "filename": None}

    # salva o arquivo temporariamente antes de passar para a thread
    suffix = os.path.splitext(audio_file.filename)[1] or ".m4a"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}{suffix}")
    audio_file.save(tmp_path)

    thread = threading.Thread(target=process_job, args=(job_id, tmp_path, titulo))
    thread.start()

    return redirect(url_for("status_page", job_id=job_id))


# ---------------------------------------------------------------
# Processamento em background
# ---------------------------------------------------------------
def transcrever_audio(caminho_audio):
    """Divide o áudio em pedaços e transcreve cada um via Groq Whisper."""
    audio = AudioSegment.from_file(caminho_audio)
    duracao_ms = len(audio)
    partes = []

    for inicio in range(0, duracao_ms, CHUNK_MS):
        pedaco = audio[inicio:inicio + CHUNK_MS]
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            pedaco.export(tmp.name, format="mp3", bitrate="64k")
            with open(tmp.name, "rb") as f:
                resultado = client.audio.transcriptions.create(
                    file=(os.path.basename(tmp.name), f.read()),
                    model="whisper-large-v3-turbo",
                    language="pt",
                    response_format="text",
                )
            partes.append(str(resultado))
        os.remove(tmp.name)

    return "\n".join(partes)


def gerar_resumo(transcricao):
    """Usa um LLM gratuito da Groq para organizar a transcrição em resumo de estudo."""
    prompt = f"""Você é um assistente que organiza transcrições de aulas de residência médica
em material de estudo, em português do Brasil.

A partir da transcrição abaixo, produza um resumo estruturado com estas seções:

1. TÓPICOS PRINCIPAIS (lista curta)
2. RESUMO POR TÓPICO (explicação organizada, com os pontos-chave de cada tópico)
3. TERMOS E DEFINIÇÕES IMPORTANTES (glossário)
4. PERGUNTAS PARA REVISÃO (5 a 10 perguntas que ajudam a fixar o conteúdo)

Seja fiel ao conteúdo da aula, não invente informação médica que não esteja na transcrição.
Escreva em português do Brasil, de forma clara e organizada.

TRANSCRIÇÃO:
{transcricao}
"""
    resposta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4000,
    )
    return resposta.choices[0].message.content


def montar_pdf(titulo, resumo, transcricao):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Corpo", parent=styles["Normal"], fontSize=11, leading=16, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="TituloAula", parent=styles["Title"], fontSize=20))

    story = [
        Paragraph(titulo, styles["TituloAula"]),
        Spacer(1, 0.5 * cm),
    ]

    for paragrafo in resumo.split("\n"):
        texto = paragrafo.strip()
        if not texto:
            story.append(Spacer(1, 0.2 * cm))
            continue
        if texto.isupper() or texto.rstrip(":").isupper():
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(f"<b>{texto}</b>", styles["Heading2"]))
        else:
            story.append(Paragraph(texto.replace("&", "&amp;"), styles["Corpo"]))

    story.append(PageBreak())
    story.append(Paragraph("Transcrição completa (apêndice)", styles["Heading2"]))
    story.append(Spacer(1, 0.3 * cm))
    for paragrafo in transcricao.split("\n"):
        texto = paragrafo.strip()
        if texto:
            story.append(Paragraph(texto.replace("&", "&amp;"), styles["Corpo"]))
            story.append(Spacer(1, 0.15 * cm))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def process_job(job_id, caminho_audio, titulo):
    try:
        transcricao = transcrever_audio(caminho_audio)
        resumo = gerar_resumo(transcricao)
        pdf_bytes = montar_pdf(titulo, resumo, transcricao)
        JOBS[job_id]["pdf_bytes"] = pdf_bytes
        JOBS[job_id]["filename"] = f"{titulo}.pdf".replace(" ", "_")
        JOBS[job_id]["status"] = "done"
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
    finally:
        if os.path.exists(caminho_audio):
            os.remove(caminho_audio)


# ---------------------------------------------------------------
# Status / download
# ---------------------------------------------------------------
@app.route("/status/<job_id>")
@login_required
def status_page(job_id):
    if job_id not in JOBS:
        return "Job não encontrado.", 404
    return render_template("status.html", job_id=job_id)


@app.route("/api/status/<job_id>")
@login_required
def api_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify({"status": job["status"], "error": job["error"]})


@app.route("/download/<job_id>")
@login_required
def download(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return "Arquivo ainda não está pronto.", 404
    return send_file(
        io.BytesIO(job["pdf_bytes"]),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=job["filename"] or "aula.pdf",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
