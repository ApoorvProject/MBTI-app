from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import pickle
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import os
import sqlite3
import re

app = Flask(__name__)

i_model_etc = pickle.load(open('i_model_etc.pkl', 'rb'))
n_model_etc = pickle.load(open('n_model_etc.pkl', 'rb'))
t_model_etc = pickle.load(open('t_model_etc.pkl', 'rb'))
j_model_etc = pickle.load(open('j_model_etc.pkl', 'rb'))

cur_dir = os.path.dirname(__file__)
db = os.path.join(cur_dir,'personality_data.sqlite')

unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
cachedStopWords = stopwords.words("english")


def cleanText(text):
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub(' +', ' ', text).lower()
    text = " ".join([lemmatiser.lemmatize(w) for w in text.split(' ') if w not in cachedStopWords])
    for t in unique_type_list:
        text = text.replace(t, "")
    return text

def sqlite_entry(path,mbti_input,results):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO livedata_db" \
              " (textual_input, personality_type, date) VALUES" \
              " (?, ?, DATETIME('now'))", (str(mbti_input), str(results)))
    conn.commit()
    conn.close()

def classify_mbti(mbti_input):
    temp_df = pd.DataFrame()
    temp_df['input'] = [mbti_input]
    temp_df['cleaned'] = temp_df['input'].apply(cleanText)
    i = int(i_model_etc.predict(temp_df['cleaned']))
    n = int(n_model_etc.predict(temp_df['cleaned']))
    t = int(t_model_etc.predict(temp_df['cleaned']))
    j = int(j_model_etc.predict(temp_df['cleaned']))

    probab = []
    probab.append(i_model_etc.predict_proba(temp_df['cleaned']))
    probab.append(n_model_etc.predict_proba(temp_df['cleaned']))
    probab.append(t_model_etc.predict_proba(temp_df['cleaned']))
    probab.append(j_model_etc.predict_proba(temp_df['cleaned']))

    probab_i = probab[0][0][1]
    probab_e = probab[0][0][0]
    probab_n = probab[1][0][1]
    probab_s = probab[1][0][0]
    probab_t = probab[2][0][1]
    probab_f = probab[2][0][0]
    probab_j = probab[3][0][1]
    probab_p = probab[3][0][0]

    complete = []
    comp_init = []
    if i == 1:
        complete.append('Introversion')
        comp_init.append('I')
    else:
        complete.append('Extroversion')
        comp_init.append('E')
    if n == 1:
        complete.append('Intuition')
        comp_init.append('N')
    else:
        complete.append('Sensing')
        comp_init.append('S')
    if t == 1:
        complete.append('Thinking')
        comp_init.append('T')
    else:
        complete.append('Feeling')
        comp_init.append('F')
    if j == 1:
        complete.append('Judging')
        comp_init.append('J')
    else:
        complete.append('Percieving')
        comp_init.append('P')
    person_string = ''
    for i in comp_init:
        person_string += i

    """print("\n/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / \n")
    print("MBTI Personality: {}".format(person_string))"""

    probab_simp = []
    for i in range(4):
        if probab[i][0][0] > probab[i][0][1]:
            append_value = probab[i][0][0]
        else:
            append_value = probab[i][0][1]
        probab_simp.append(append_value)
    '''
    results = pd.DataFrame({'Characteristic':complete,'Probability':probab_simp})
    print("Your MBTI characteristics are: {}".format(person_string_comp))
    results'''
    #results = pd.DataFrame({'Characteristic': complete, 'Abbreviation': comp_init, 'Probability': probab_simp})
    results = person_string
    #percent introversion=
    #percent intuitive=
    return results, probab_i, probab_e, probab_n, probab_s, probab_t, probab_f, probab_j, probab_p



class MBTIForm(Form):
    mbti_text = TextAreaField('', [validators.DataRequired(),
                                  validators.length(min=15)], render_kw={"placeholder":"Enter/Type something written by you, preferably more than a 100 words"})

@app.route('/')
def index():
    form = MBTIForm(request.form)
    return render_template('mbti_input.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = MBTIForm(request.form)
    if request.method == 'POST' and form.validate():
        raw_text = request.form['mbti_text']
        results_df, probab_i, probab_e, probab_n, probab_s, probab_t, probab_f, probab_j, probab_p = classify_mbti(raw_text)
        return render_template('results.html',
                               content=raw_text,
                               results=results_df,
                               i_probab=probab_i,
                               e_probab=probab_e,
                               n_probab=probab_n,
                               s_probab=probab_s,
                               t_probab=probab_t,
                               f_probab=probab_f,
                               j_probab=probab_j,
                               p_probab=probab_p)
    return render_template('mbti_input.html', form=form)

@app.route('/thanks', methods=['POST'])
def db_entry():
    db_entry = request.form['db_entry_button']
    content = request.form['content']
    final_results = request.form['results']

    sqlite_entry(db, content, final_results)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)