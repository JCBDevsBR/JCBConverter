from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for
import os
import shutil
import uuid
import subprocess
import sys
import logging

import librosa
import soundfile as sf
from pydub import AudioSegment
from yt_dlp import YoutubeDL

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Configurações
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
STEMS_FOLDER = "stems_tmp"
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'aac', 'ogg'}

for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER, STEMS_FOLDER):
    os.makedirs(folder, exist_ok=True)

# Utilitários ------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_audio_from_url(url: str) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(UPLOAD_FOLDER, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base = ydl.prepare_filename(info)
    return os.path.splitext(base)[0] + ".mp3"

def ensure_wav(path_in: str) -> str:
    if path_in.lower().endswith(".wav"):
        return path_in
    wav_path = os.path.splitext(path_in)[0] + ".wav"
    AudioSegment.from_file(path_in).export(wav_path, format="wav")
    return wav_path

def spleeter_accompaniment(wav_path: str):
    job = uuid.uuid4().hex[:8]
    out_dir = os.path.join(STEMS_FOLDER, job)
    cmd = [
        sys.executable, "-m", "spleeter", "separate",
        wav_path, "-p", "spleeter:2stems", "-o", out_dir
    ]
    subprocess.run(cmd, check=True)
    base = os.path.splitext(os.path.basename(wav_path))[0]
    acc_path = os.path.join(out_dir, base, "accompaniment.wav")
    if not os.path.exists(acc_path):
        raise FileNotFoundError("accompaniment.wav não gerado pelo Spleeter.")
    return acc_path, out_dir

# Rotas ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_audio():
    try:
        # 1. Entrada: arquivo OU URL
        file = request.files.get("audio")
        url  = request.form.get("audio_url")

        if file and file.filename:
            # Validar extensão
            if not allowed_file(file.filename):
                return "Formato de arquivo não suportado", 400
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

        elif url:
            try:
                input_path = download_audio_from_url(url)
            except Exception as e:
                app.logger.error(f"Erro no download: {e}")
                return f"Erro ao baixar áudio: {e}", 400

        else:
            return "Nenhum arquivo ou URL fornecido", 400

        # 2. Converter para WAV se necessário
        wav_path = ensure_wav(input_path)

        # 3. Remover vocais (opcional)
        strip = request.form.get("strip_vocals") == "yes"
        acc_path, spleeter_folder = (wav_path, None)
        if strip:
            acc_path, spleeter_folder = spleeter_accompaniment(wav_path)

        # 4. Carregar áudio
        y, sr = librosa.load(acc_path, sr=None)

        # 5. Ajustar pitch e tempo
        try:
            n_steps = float(request.form.get("pitch_shift", 0))
            tempo   = float(request.form.get("tempo_factor", 1))
        except ValueError:
            return "Parâmetros inválidos", 400

        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        y = librosa.effects.time_stretch(y, rate=tempo)

        # 6. Salvar saída no formato desejado
        output_format = request.form.get("output_format", "wav").lower()
        base = os.path.splitext(os.path.basename(wav_path))[0]
        out_name = f"processed_{base}.{output_format}"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)

        if output_format == "wav":
            sf.write(out_path, y, sr)
        else:
            tmp = os.path.join(OUTPUT_FOLDER, f"tmp_{uuid.uuid4().hex}.wav")
            sf.write(tmp, y, sr)
            AudioSegment.from_wav(tmp).export(
                out_path, format="mp3", bitrate="192k"
            )
            os.remove(tmp)

        # 7. Limpeza
        if spleeter_folder:
            shutil.rmtree(spleeter_folder, ignore_errors=True)
        if os.path.exists(input_path):
            os.remove(input_path)

        # 8. Resposta: JSON para AJAX, senão redirect normal
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            # Retorna {"file":"processed_....wav"} com Content-Type: application/json :contentReference[oaicite:0]{index=0}
            return jsonify(file=out_name)
        # Redireciona navegador padrão para /preview?file=...
        return redirect(url_for("preview", file=out_name))

    except Exception as e:
        app.logger.error(f"Erro no processamento: {e}")
        return f"Erro interno: {e}", 500


@app.route("/preview")
def preview():
    filename = request.args.get("file")
    if not filename:
        return redirect(url_for('index'))
    return render_template("preview.html", audio_file=filename)

@app.route("/preview/<filename>")
def preview_file(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return "Arquivo não encontrado", 404
    return send_file(path, conditional=True)

@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return "Arquivo não encontrado", 404
    return send_file(path, as_attachment=True)

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    return "Erro interno do servidor", 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)