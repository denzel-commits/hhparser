import time
import json
import requests
import tqdm
from IPython.display import clear_output

import pandas as pd
from collections import Counter
from wordcloud import WordCloud

from config import HH_VACANCIES_ENDPOINT, PROJECT_DATA_PATH


def dump_json(obj, filename):
    """Saves JSON file on disk"""
    with open(filename, "w", encoding="UTF-8") as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=4)


def get_vacancies(text="python", experience=None, employment=None, schedule=None):
    """Downloads data form HH API"""
    params = {
        "per_page": 100,
        "page": 0,
        "period": 30,
        "text": text,
        "experience": experience,
        "employment": employment,
        "schedule": schedule,
    }

    res = requests.get(HH_VACANCIES_ENDPOINT, params=params)

    if not res.ok:
        print("Error", res)
        return []

    vacancies = res.json()["items"]
    pages = res.json()["pages"]

    for page in tqdm.trange(1, pages):
        params["page"] = page
        res = requests.get(HH_VACANCIES_ENDPOINT, params=params)
        if res.ok:
            response_json = res.json()
            vacancies.extend(response_json["items"])
        else:
            print(res)

    dump_json(vacancies, PROJECT_DATA_PATH + "vacancies.json")

    return vacancies


def get_full_descriptions(vacancies):
    """Gets vacancies full description (works 20 mins)"""
    vacancies_full = []
    for entry in tqdm.tqdm(vacancies):
        vacancy_id = entry["id"]
        description = requests.get("{}/{}".format(HH_VACANCIES_ENDPOINT, vacancy_id))
        vacancies_full.append(description.json())
        time.sleep(0.2)
        clear_output()

    dump_json(vacancies_full, PROJECT_DATA_PATH + "vacancies_full.json")

    return vacancies_full


def load_from_google_drive(file_id, filename):
    """Gets full description from google cache file"""
    res = requests.get(f"https://drive.google.com/uc?export=view&id={file_id}")
    data = res.json()
    dump_json(data, filename)
    return data


vacancies = get_vacancies(employment="full", experience="between1And3")
print("Загружено", len(vacancies), "вакансий")

# vacancies_full = get_full_descriptions(vacancies)
vacancies_full = load_from_google_drive("1d2NfxfM2n48m5WS6oCCc3rcQ4hdnTQ1v", PROJECT_DATA_PATH + "vacancies_full.json")

all_skills = []
for vacancy in vacancies_full:
    for skill in vacancy["key_skills"]:
        all_skills.append((skill["name"]))

frequencies = Counter(all_skills)

pd.DataFrame(vacancies).to_excel(PROJECT_DATA_PATH + "vacancy_list.xlsx")
pd.DataFrame(vacancies_full).to_excel(PROJECT_DATA_PATH + "full_vacancies.xlsx")

print("Топ навыков Python разработчика:")
cloud = WordCloud(background_color="white")
wc_skills = cloud.generate_from_frequencies(frequencies=frequencies).to_image()
wc_skills.show()
wc_skills.save(PROJECT_DATA_PATH + 'word_cloud_skills.png')
