from flask import Flask, request, jsonify, render_template
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import load_model
import json
import random
import nltk
import os  # Added to get the file paths

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

import pickle

# Load necessary files
model_path = os.path.join('static', 'chatassistant_model.h5')
intents_path = os.path.join('static', 'intents.json')
words_path = os.path.join('static', 'words.pkl')
classes_path = os.path.join('static', 'classes.pkl')

model = load_model(model_path)
intents = json.loads(open(intents_path).read())
words = pickle.load(open(words_path, 'rb'))
classes = pickle.load(open(classes_path, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    ints = predict_class(user_message, model)
    res = getResponse(ints, intents)
    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(debug=True)
