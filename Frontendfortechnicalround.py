import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import tkinter as tk
from tkinter import *

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load necessary files
model = load_model('static/chatassistant_model.h5')
intents = json.loads(open('static/intents.json').read())
words = pickle.load(open('static/words.pkl', 'rb'))
classes = pickle.load(open('static/classes.pkl', 'rb'))

# Function to clean up sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert sentence into bag of words
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

# Function to predict class
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

# Function to get response based on intent
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Function to generate chatbot response
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res

# GUI setup using Tkinter
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Auge", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "PrinceSolutions: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

# Setup main application window
base = Tk()
base.title("Prince Solutions Chat Assistant")
base.geometry("400x460")
base.resizable(width=FALSE, height=FALSE)

# Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Auge")
ChatLog.config(state=DISABLED)

# Scrollbar for chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Send button
SendButton = Button(base, font=("Auge", 12, 'bold'), text="Send", width="12", height="2",
                    bd=0, bg="#5500C2", activebackground="#9C4EFF", fg='#ffffff',
                    command=send)

# Entry box for user input
EntryBox = Text(base, bd=0, bg="white", width="29", height="2", font="Arial", borderwidth=2)

# Place all components on the window
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=5, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=50, width=265)
SendButton.place(x=6, y=401, height=50)

# Start the application
base.mainloop()
