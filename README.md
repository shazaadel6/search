!pip install gradio langdetect sentence-transformers torch SpeechRecognition pydub -q
import gradio as gr
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import speech_recognition as sr
import io
from pydub import AudioSegment

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

students_data = pd.DataFrame([
    {"name": "أحمد علي", "national_id": "1234567890", "grade": "الثالث الثانوي", "disease": "السكر"},
    {"name": "محمد حسن", "national_id": "9876543210", "grade": "الثاني الإعدادي", "disease": "الضغط"},
    {"name": "Sara Ahmed", "national_id": "5555555555", "grade": "Grade 10", "disease": "Diabetes"},
    {"name": "Ali Hassan", "national_id": "1111222233", "grade": "Grade 12", "disease": "None"},
    {"name": "مريم يوسف", "national_id": "9999999999", "grade": "الأول الثانوي", "disease": "None"},
])

def smart_search(query):
    try:
        lang = detect(query)
    except LangDetectException:
        lang = "en"
    query_emb = model.encode([query], convert_to_numpy=True)
    student_texts = students_data.apply(lambda row: f"{row['name']} {row['national_id']} {row['grade']} {row['disease']}", axis=1)
    text_embs = model.encode(student_texts.tolist(), convert_to_numpy=True)
    cosine_scores = np.dot(query_emb, text_embs.T) / (np.linalg.norm(query_emb) * np.linalg.norm(text_embs, axis=1))
    top_idx = np.argmax(cosine_scores)
    if cosine_scores[0][top_idx] < 0.4:
        return " لم يتم العثور على نتائج قريبة. جرّب كتابة السؤال بطريقة مختلفة."
    result = students_data.iloc[top_idx]
    return f"الاسم: {result['name']}\n الرقم القومي: {result['national_id']}\n الصف: {result['grade']}\n الحالة الصحية: {result['disease']}"

def speech_to_text(audio):
    if audio is None:
        return "لم يتم إدخال صوت."
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(io.BytesIO(audio), format="wav")
    audio.export("temp.wav", format="wav")
    with sr.AudioFile("temp.wav") as source:
        recorded_audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(recorded_audio, language="ar-AR")
        return text
    except sr.UnknownValueError:
        return "لم أستطع فهم الصوت، حاول مرة أخرى."

with gr.Blocks(title=" Student Search Assistant") as demo:
    gr.Markdown("##  مساعد البحث عن الطلاب (عربي + إنجليزي)")
    with gr.Tab(" بحث نصي"):
        query = gr.Textbox(label="اكتب اسم الطالب أو الرقم القومي أو المرض")
        text_output = gr.Textbox(label="النتيجة")
        text_button = gr.Button("ابحث")
        text_button.click(fn=smart_search, inputs=query, outputs=text_output)
    with gr.Tab(" بحث صوتي"):
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="سجّل صوتك هنا")
        audio_text = gr.Textbox(label="النص المستخرج من الصوت")
        convert_button = gr.Button(" تحويل الصوت إلى نص")
        convert_button.click(fn=speech_to_text, inputs=audio_input, outputs=audio_text)
        result_button = gr.Button(" ابحث بناءً على الصوت")
        result_button.click(fn=smart_search, inputs=audio_text, outputs=text_output)
demo.launch()
