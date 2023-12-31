# WhileTrue_Credit_ratings
## Анализатор текстовых пресс-релизов
На основании исторических пресс-релизов
кредитных рейтинговых агентств необходимо
построить интерпретируемую ML-модель,
устанавливающую взаимосвязь между текстом
пресс-релиза и присвоенным кредитным
рейтингом организации с учетом
методологических особенностей оценки рейтинга.
Целевое решение (MVP): ML-модель
должна не просто устанавливать соответствие
текста пресс-релиза кредитному рейтингу, но
также и выделять ключевые конструкции
в тексте, соответствующие присвоенному кредитному рейтингу.

## ML-модель

Регреесия с дообучением модели bert от ai-forever. Определяет рейтинг по каждому предложению. Без использования парсинга всего документа.

https://github.com/Demontego/WhileTrue_Credit_ratings/assets/36605450/5bf3dd86-2ee6-4294-81f6-ee9dceb425d9

Порядок запуска:
1. Установить anaconda и jupyter notebook
2. Установить requirements.txt и pandas(for excel)
3. Обучить модель в ноутбуке
4. Сохранить последний чекпоинт в корневую директорию с названием molbert
5. Запустить app.py
   
## Простое решение при помощи чат-гпт

Подаем на вход такой promt:
Ты кредитное рейтинговое агентство АО «Эксперт РА», которое присваивает рейтинги кредитоспособности нефинансовым компаниям на основе пресс-релизов
Оценки которые ты можешь ставить:
AAA - Объект рейтинга характеризуется максимальным уровнем кредитоспособности устойчивости.
AAA - Объект рейтинга характеризуется максимальным уровнем кредитоспособности устойчивости.
AA - Высокий уровень кредитоспособности устойчивости, который лишь незначительно ниже, чем у объектов рейтинга в рейтинговой категории AAA.
А - Умеренно высокий уровень кредитоспособности устойчивости, однако присутствует некоторая чувствительность к воздействию негативных изменений экономической конъюнктуры.
BBB - Умеренный уровень кредитоспособности устойчивости, при этом присутствует более высокая чувствительность к воздействию негативных изменений экономической конъюнктуры, чем у объектов рейтинга в рейтинговой категории A.
BB - Умеренно низкий уровень кредитоспособности устойчивости. Присутствует высокая чувствительность к воздействию негативных изменений экономической конъюнктуры.
B - Низкий уровень кредитоспособности. В настоящее время сохраняется возможность исполнения финансовых обязательств в срок и в полном объеме, однако при этом запас прочности ограничен
CCC - Очень низкий уровень кредитоспособности. Существует значительная вероятность невыполнения объектом рейтинга своих финансовых обязательств уже в краткосрочной перспективе.
CC - Очень низкий уровень кредитоспособности устойчивости. Существует повышенная вероятность невыполнения объектом рейтинга своих финансовых обязательств уже в краткосрочной перспективе.
C - Очень низкий уровень кредитоспособности устойчивости. Существует очень высокая вероятность невыполнения объектом рейтинга своих финансовых обязательств уже в краткосрочной перспективе.

Пример анализа: https://chat.openai.com/share/9729eef1-1979-4497-b806-53befbe1099a (заходите под VPN)
