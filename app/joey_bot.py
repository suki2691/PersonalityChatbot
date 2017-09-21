from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import re
import codecs
import string

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

class ReusableForm(Form):
    sentence = TextField('Input sentence:', validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)

    # print(form.errors)
    if request.method == 'POST':
        sentence=request.form['sentence']
        print(sentence)

        if form.validate():
            # Save the comment here.
            def vectorization(text,chars):
                char_indices = dict((c, i) for i, c in enumerate(chars))
                indices_char = dict((i, c) for i, c in enumerate(chars))

                # cut the text in semi-redundant sequences of maxlen characters
                maxlen = 40
                step = 3
                sentences = []
                next_chars = []
                for i in range(0, len(text) - maxlen, step):
                    sentences.append(text[i: i + maxlen])
                    next_chars.append(text[i + maxlen])
                return maxlen, char_indices, indices_char

            def sample(preds, temperature=1.0):
                '''helper function to sample an index from a probability array'''

                preds = np.asarray(preds).astype('float64')
                preds = np.log(preds) / temperature
                exp_preds = np.exp(preds)
                preds = exp_preds / np.sum(exp_preds)
                probas = np.random.multinomial(1, preds, 1)
                return np.argmax(probas)

            def text_gen(sentence, model, chars, char_indices, indices_char, maxlen=40):
                '''Function prints dialogues, predicted one character (letter) at a time'''

                generated = ''
                # generated += sentence
                for i in range(100):
                    x = np.zeros((1, maxlen, len(chars)))

                    for t, char in enumerate(sentence):
                        x[0, t, char_indices[char]] = 1.


                    preds = model.predict(x, verbose=0)[0]
                    next_index = sample(preds, 0.8)
                    next_char = indices_char[next_index]

                    if next_char in ['!','?','.']:
                        generated += next_char
                        break
                    else:
                        generated += next_char
                        sentence = sentence[1:] + next_char

                        sys.stdout.write(next_char)
                        sys.stdout.flush()
                print()
                return generated

            path = "joey_dialogues.txt"
            with codecs.open(path, encoding='utf-8') as f:
                text = f.read().lower()
                text = ''.join(filter(string.printable.__contains__,text))
                text = re.sub(r'\n',' ',text.strip())
                text = text.strip()

            chars = sorted(list(set(text)))

            maxlen, char_indices, indices_char = vectorization(text, chars)

            left = Sequential()
            left.add(LSTM(256, input_shape=(maxlen, len(chars)),return_sequences=True))
            left.add(LSTM(128, input_shape=(maxlen, len(chars))))
            left.add(Dense(len(chars)))
            left.add(Activation('softmax'))
            filename = 'weights-improvement-00-0.1512-joey_new.hdf5'
            left.load_weights(filename)
            optimizer = RMSprop(lr=0.001)
            left.compile(loss='categorical_crossentropy', optimizer=optimizer)

            if len(sentence) < 40:
                sentence = (40-len(sentence))*' ' + sentence

            generate = text_gen(sentence, left, chars, char_indices, indices_char)
            flash(generate)
        else:
            flash('All the form fields are required. ')
    print(form.errors)
    return render_template('app.html', form=form)

if __name__ == "__main__":

    app.run()
