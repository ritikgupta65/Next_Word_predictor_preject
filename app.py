from flask import Flask, request, render_template , redirect , url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

try:
    model = tf.keras.models.load_model('model.h5', compile=False)
except Exception as e:
    print(f"Error loading model: {e}")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST' ,'GET'])
def submit():
    text = request.form['text']
    num_words = int(request.form['no_of_words'])
    
    for i in range(num_words):
        tokenized_text = tokenizer.texts_to_sequences([text])[0]
        padded_text = pad_sequences([tokenized_text], maxlen=62, padding='pre')

        prediction = model.predict(padded_text)
        next_word_index = np.argmax(prediction)

        # next_word = reverse_word_index.get(next_word_index)
    
      
        # if next_word:
        #    text += " " + next_word 
   
        for word,index in tokenizer.word_index.items():
          if index == next_word_index:
            text = text + " " + word
       

    output = text 
    
    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)