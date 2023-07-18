import os
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
import smtplib
from email.message import EmailMessage
import schedule
from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter
from sklearn.feature_extraction.text import TfidfVectorizer

from config import EMAIL_CREDENTIALS

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
            "text": 'python',
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

    dump_json(vacancies, 'vacancies.json')

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

    dump_json(vacancies_full, 'vacancies_full.json')

    return vacancies_full


def load_from_google_drive(file_id, filename):
    """Функция для загрузки уже скаченного файла вместо get_full_descriptions"""
    url = f"https://drive.google.com/uc?export=view&id={file_id}"
    res = requests.get(url)
    data = res.json()
    dump_json(data, filename)
    return data


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

    dump_json(preprocessed, 'preprocessed.json')

    return preprocessed


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

# Код 3-го дня: автоматизация email-рассылки по расписанию

def generate_digest(text="python", experience=None,
                    employment=None, schedule=None,
                    load_presaved=False):
    """Эта функция объединяет в себе все предыдущие и с нуля генерирует все нужные
    для отправки email файлы.
    Если параметр load_presaved=True, остальные параметры будут проигнорированы и
    загрузится предсохраненная версия с диска.
    """
    if load_presaved:
        vacancies_full = load_from_google_drive(
            '1d2NfxfM2n48m5WS6oCCc3rcQ4hdnTQ1v', 'vacancies_full.json')
    else:
        vacancies = get_vacancies(text, experience, employment, schedule)
        print('Загружено', len(vacancies), 'вакансий')
        vacancies_full = get_full_descriptions(vacancies)

    all_skills = []
    for vacancy in vacancies_full:
        for skill in vacancy['key_skills']:
            all_skills.append(skill['name'])

    frequencies = Counter(all_skills)

    pd.DataFrame(vacancies_full).to_excel('Подробное описание вакансий.xlsx')

    print('Топ навыков Python-разработчика:')
    cloud = WordCloud(background_color="white")
    wc_skills = cloud.generate_from_frequencies(frequencies).to_image()
    wc_skills.show()
    wc_skills.save('word_cloud_skills.png')

    vacancies_df = pd.DataFrame(vacancies_full)

    if load_presaved:
        preprocessed = load_from_google_drive(
            '14dcZnE_XvVeCgWCDgJSZiHgYJQm-TYvD', 'preprocessed.json')
    else:
        preprocessed = preprocess_all(vacancies_df['description'])

    frequencies = get_tf_idf_weights(preprocessed)

    print('\nКлючевые слова в описании вакансий:')
    cloud = WordCloud(background_color="white")
    wc_descriptions = cloud.generate_from_frequencies(frequencies).to_image()
    wc_descriptions.show()
    wc_descriptions.save('word_cloud_descriptions.png')


def send_email(to, sender, password):
    """Эта функция отправляет email"""
    msg = EmailMessage()
    msg['Subject'] = 'Тренды вакансий'
    msg['From'] = sender
    msg['To'] = to

    msg.set_content('Персональная подборка вакансий')

    # Вставляем HTML
    html_content = '''
    Добрый день, дорогой подписчик!<br>
    <br>
    Присылаем дайджест с трендами в вакансиях для Python-разработчиков!<br>

    Основные навыки:<br>
    <img src="cid:0" style="width:200px"><br>
    <br>
    Самые популярные ключевые слова в описании вакансий:<br>

    <img src="cid:1" style="width:200px"><br>
    <br>
    Во вложениях&nbsp;&mdash; подробная информация о вакансиях.<br>
    <br>
    <br>
    С уважением,<br>
    Python-разработчик 🐍
    '''
    msg.add_alternative(html_content, subtype='html')

    # Считываем файлы в бинарном виде и прикрепляем к письму

    with open('word_cloud_skills.png', 'rb') as f:
            file_data = f.read()
            msg.get_payload()[-1].add_related(
                file_data, 'word_cloud_skills.png', 'png', cid='<0>')

    with open('word_cloud_descriptions.png', 'rb') as f:
            file_data = f.read()
            msg.get_payload()[-1].add_related(
                file_data, 'word_cloud_descriptions.png', 'png', cid='<1>')

    with open('Подробное описание вакансий.xlsx', 'rb') as f:
            file_data = f.read()
            msg.add_attachment(
                file_data, maintype="application",
                subtype="xlsx", filename='Подробное описание вакансий.xlsx')

    with smtplib.SMTP_SSL('smtp.yandex.ru', 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)


# Прочитаем данные c логином и паролем из файла
# cred_filename = 'credentials.json'
#
# if os.path.exists(cred_filename):
#     with open(cred_filename) as f:
#         credentials = json.load(f)
# else:
#     raise FileNotFoundError(
#         f'Файл с реквизитами к почтовому ящику {cred_filename} не найден,\n'
#         'пожалуйста, воспользуйтесь инструкцией выше')

credentials = EMAIL_CREDENTIALS


def job_send_digest():
    """Эта функция вызывает последовательно генерацию и отправку отчета.
    Так как она запускается планировщиком по расписанию, она не должна иметь параметров.
    """
    # Вариант набора параметров (суммарно отрабатывает примерно 12 минут вместо 50)
    generate_digest(
        # Тут можно настроить параметры (не забудьте отключить load_presaved)
        text="python",
        experience='noExperience',
        employment=['full', 'part'],
        schedule=['fullDay', 'remote'],
        # Чтобы загружать новые данные по найденным параметрам,
        # замените следующую строчку на load_presaved=False,
        # а чтобы получить предсохраненные данные - на load_presaved=True
        load_presaved=False
        )

    # Если хотим разослать одни и те же данные нескольким пользователям, нужно создать
    # список адресов и вызывать функцию send_email(to, ...) в цикле
    send_email(credentials['email'],
               credentials['email'],
               credentials['password'])


schedule.clear()
schedule.every(1).day.do(job_send_digest)
schedule.run_all()


# Этот цикл будет повторяться бесконечно, пока ячейка не будет остановлена вручную
# (это вызовет KeyboardInterrupt - просто переходим к выполнению следующей ячейки)
while True:
    schedule.run_pending()
    time.sleep(1)