import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random

# Inisialisasi Lemmatizer
lemmatizer = WordNetLemmatizer()

# Memuat file-file penting yang sudah dibuat sebelumnya
print("Sedang memuat model dan data...")
model = load_model('chatbot_model.h5')
intents = json.loads(open('penyakit_ikan.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    # Tokenisasi (pemecahan kalimat menjadi kata)
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatisasi (mengubah kata ke bentuk dasar)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Mengubah kalimat menjadi Bag of Words (angka 0 dan 1)
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, model):
    # Prediksi intent user
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    
    # Ambil probabilitas tertinggi (threshold 0.25)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Urutkan dari yang paling mungkin (probabilitas tinggi)
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    # Mengambil teks jawaban berdasarkan tag intent yang diprediksi
    if not ints:
        return "Maaf, saya tidak mengerti gejalanya. Bisa dijelaskan lebih detail?"
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# --- AREA UTAMA CHATBOT ---
print("="*50)
print("BOT PAKAR PENYAKIT IKAN SIAP! (Ketik 'keluar' untuk berhenti)")
print("="*50)

while True:
    message = input("\nAnda: ")
    if message.lower() == "keluar":
        break
    
    # Prediksi
    ints = predict_class(message, model)
    # Ambil Jawaban
    res = get_response(ints, intents)
    
    print("Bot :", res)