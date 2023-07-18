import time
import json
import requests
from IPython.display import display, clear_output
import tqdm
import tqdm.notebook

import bs4
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter
from sklearn.feature_extraction.text import TfidfVectorizer

from config import PROJECT_MINING_PATH


# Код 1-го дня: загрузка данных из интернета

def dump_json(obj, filename):
    """Функция сохранения JSON-файла на диск"""
    with open(filename, 'w', encoding='UTF-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def get_vacancies(text="python",
                  experience=None, employment=None, schedule=None):
    """Функция для скачивания данных по API HeadHunter"""
    params = {
            "per_page": 100,
            "page": 0,
            "period": 30,
            "text": text,
            "experience": experience,
            "employment": employment,
            "schedule": schedule,
        }

    res = requests.get("https://api.hh.ru/vacancies", params=params)
    if not res.ok:
        print('Error:', res)
    vacancies = res.json()["items"]
    pages = res.json()['pages']

    for page in tqdm.trange(1, pages):
        params['page'] = page
        res = requests.get("https://api.hh.ru/vacancies", params=params)
        if res.ok:
            response_json = res.json()
            vacancies.extend(response_json["items"])
        else:
            print(res)

    dump_json(vacancies, PROJECT_MINING_PATH + 'vacancies.json')

    return vacancies


def get_full_descriptions(vacancies):
    """Функция для скачивания полного описания вакансий (работает 20 минут!)"""
    vacancies_full = []
    for entry in tqdm.tqdm(vacancies):
        vacancy_id = entry['id']
        description = requests.get(f"https://api.hh.ru/vacancies/{vacancy_id}")
        vacancies_full.append(description.json())
        print(description.json())
        time.sleep(0.2) # Этот таймаут нужен, чтобы hh не начал запрашивать капчу
        clear_output()

    dump_json(vacancies_full, PROJECT_MINING_PATH + "vacancies_full.json")

    return vacancies_full


def load_from_google_drive(file_id, filename):
    """Функция для загрузки уже скаченного файла вместо get_full_descriptions"""
    url = f"https://drive.google.com/uc?export=view&id={file_id}"
    res = requests.get(url)
    data = res.json()
    dump_json(data, filename)
    return data


vacancies = get_vacancies()
print('Загружено', len(vacancies), 'вакансий')


# vacancies_full = get_full_descriptions(vacancies)  # Выполняется ≈20 мин
vacancies_full = load_from_google_drive('1d2NfxfM2n48m5WS6oCCc3rcQ4hdnTQ1v', PROJECT_MINING_PATH + 'vacancies_full.json')

all_skills = []
for vacancy in vacancies_full:
    for skill in vacancy['key_skills']:
        all_skills.append(skill['name'])

frequencies = Counter(all_skills)

pd.DataFrame(vacancies).to_excel(PROJECT_MINING_PATH + "vacancy_list.xlsx")
pd.DataFrame(vacancies_full).to_excel(PROJECT_MINING_PATH + "full_vacancies.xlsx")

print('Топ навыков Python-разработчика:')
cloud = WordCloud(background_color="white")
wc_skills = cloud.generate_from_frequencies(frequencies).to_image()
wc_skills.show()
wc_skills.save(PROJECT_MINING_PATH + 'word_cloud_skills.png')


# Код 2-го дня: Data Mining - выделение ключевых слов

def preprocess(text):
    """Функция для предварительной обработки текста одной вакансии:
    '<p><strong>Кого мы ищем:</strong><br/>Junior Backend разработчика, готового работать в команде.</p> <p><strong>'
     ↓ ↓ ↓
    'искать junior backend разработчик готовый работать команда'
    """
    parsed_html = bs4.BeautifulSoup(text)
    text = parsed_html.text # удалили тэги

    morph_vocab = MorphVocab()
    segmenter = Segmenter()
    embedding = NewsEmbedding()
    morph_tagger = NewsMorphTagger(embedding)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    words = []

    for token in doc.tokens:
        # Если часть речи не входит в список: [знак пунктуации, предлог, союз, местоимение], выполняем:
        if token.pos not in ['PUNCT', 'ADP', 'CCONJ', 'PRON']:
            # Преобразуем к нормальной форме 'способов' -> 'способ'
            token.lemmatize(morph_vocab)
            # Добавляем в общий список
            words.append(token.lemma)

    # Объединяем список элементов в одну строку через пробел
    # ['обязанность', 'писать',  'код'] -> 'обязанность писать код'
    line = ' '.join(words)

    return line

def preprocess_all(document_collection):
    """Функция для обработки всех вакансий. На вход функция получает список с описаниями.
    Работает до получаса!
    """
    preprocessed = []
    for vacancy in tqdm.tqdm(document_collection):
        preprocessed.append(preprocess(vacancy))

    dump_json(preprocessed, PROJECT_MINING_PATH + 'preprocessed.json')

    return preprocessed

vacancies_df = pd.DataFrame(vacancies_full)
# Этот код выполняется примерно полчаса
# preprocessed = preprocess_all(vacancies_df['description'])

# Вместо этого загрузим обработанные тексты с диска
preprocessed = load_from_google_drive('14dcZnE_XvVeCgWCDgJSZiHgYJQm-TYvD', PROJECT_MINING_PATH + 'preprocessed.json')


def get_tf_idf_weights(preprocessed):
    """Эта функция получает на вход подготовленные тексты вида
    'искать junior backend разработчик готовый работать команда'
    и составляет по ним словарь весов ключевых слов:
    {'искать': 0.54, 'junior': 0.73, ...}
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    #  Обучаем объект векторизатора (функции, кодирующий текст в виде последовательностей чисел)
    vectorizer.fit(preprocessed)

    tf_idf_words = vectorizer.get_feature_names_out()
    tf_idf_table = vectorizer.transform(preprocessed).toarray()
    weights = tf_idf_table.sum(axis=0)
    indices_order = weights.argsort()[::-1]

    tf_idf_words[indices_order]

    frequencies = dict(zip(tf_idf_words, weights))

    return frequencies

frequencies = get_tf_idf_weights(preprocessed)

print('\nКлючевые слова в описании вакансий:')
cloud = WordCloud(background_color="white")
wc_descriptions = cloud.generate_from_frequencies(frequencies).to_image()
wc_descriptions.show()
wc_descriptions.save(PROJECT_MINING_PATH + 'word_cloud_descriptions.png')
