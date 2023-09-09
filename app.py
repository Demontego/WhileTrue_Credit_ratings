import re
import pickle
import gradio as gr
from simpletransformers.classification import ClassificationModel
from sentence_splitter import SentenceSplitter


splitter = SentenceSplitter(language='ru')
model = ClassificationModel('bert','molbert')
with open('non_informations.pkl', 'rb') as handle:
    exclude_data = pickle.load(handle)


def clean_text(text):
    # создаем регулярное выражение для удаления лишних символов
    regular = r'[\*+\#+\№\"\-+\+\=+\?+\&\^\.+\;\,+\>+\(\)\/+\:\\+]'
    # регулярное выражение для замены ссылки на "URL"
    regular_url = r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)'
    # удаляем лишние символы
    text = re.sub(regular, '', text)
    # заменяем ссылки на "URL"
    text = re.sub(regular_url, r'URL', text)
    # удаляем лишние пробелы
    text = re.sub(r'[\n\t\s+]', ' ', text)
    # возвращаем очищенные данные
    return text


dict_ratings_detail = {
    'A': ((6.33+5.92)/2, 6.33),
    'A+': ((6.75+6.33)/2, 6.75),
    'A-': ((5.92+5.54)/2, 5.92),
    'AA': ((7.68+7.20)/2, 7.68),
    'AA+': ((8.22+7.68)/2, 8.22),
    'AA-': ((7.20+6.75)/2, 7.20),
    'AAA': ((10.0+8.22)/2, 10.0),
    'B': ((2.95+2.56)/2, 2.95),
    'B+': ((3.33+2.95)/2, 3.33),
    'B-': ((2.56+2.16)/2, 2.56),
    'BB': ((4.06+3.7)/2, 4.06),
    'BB+': ((4.42+4.06)/2, 4.42),
    'BB-': ((3.70+3.33)/2, 3.70),
    'BBB+': ((5.54+5.16)/2, 5.54),
    'BBB-': ((4.79+4.42)/2, 4.79),
    'BBB': ((5.16+4.79)/2, 5.16),
    'C': (2.16/2, 2.16),
}

dict_ratings = {
    'A': 6.75,
    'AA': 8.22,
    'AAA': 10.0,
    'B': 3.33,
    'BB': 4.42,
    'BBB': 5.54,
    'C': 2.16,
}

ratings = sorted(dict_ratings.items(), key=lambda x: x[1])
ratings_detail = sorted(dict_ratings_detail.items(), key=lambda x: x[1][1])

def mapping_ratings(x):
    for label, (_, r) in ratings_detail:
        if x < r:
            return label

def text_analysis(text):
    sents = [clean_text(i) for i in splitter.split(text) if len(i.split()) > 7 and i not in exclude_data]
    outputs, _ = model.predict(sents)
    rating = outputs.mean()
    answer = 'AAA'
    detail_answer = 'AAA'
    pos_tokens = [(i, mapping_ratings(k)) for i, k in zip(sents, outputs)]

    for label, r in ratings:
        if rating < r:
            answer = label
            break
    detail_answer = mapping_ratings(rating)
    
    result = {
        'Уровень кредитного рейтинга': answer,
        'Детализированный уровень кредитного рейтинга': detail_answer,
        'Среднее значение полученных баллов': rating,
    }

    return result, [(i, k) for i,k in pos_tokens if k.startswith(answer)]


demo = gr.Interface(
    text_analysis,
    gr.Textbox(placeholder="Enter text here..."),
    ["json", 
     gr.HighlightedText()],
    examples=[
        ["«Эксперт РА» подтвердил рейтинг компании «Мостотрест»..."],
        ["В то же время, устойчивая база государственных..."],
    ],
)



if __name__ == "__main__":
    demo.queue(max_size=4)
    demo.launch(inbrowser=True, height=1200, share=True)
