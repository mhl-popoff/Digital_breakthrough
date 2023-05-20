# Импортируем библиотеки
import pandas as pd                # Импорт библиотеки pandas для работы с данными в формате DataFrame
import numpy as np                 # Импорт библиотеки numpy для работы с массивами и вычислений
import re                          # Импорт модуля re для работы с регулярными выражениями
import pymorphy2                   # Импорт библиотеки pymorphy2 для морфологического анализа слов
import nltk                        # Импорт библиотеки nltk для обработки текста
import matplotlib.pyplot as plt    # Импорт библиотеки matplotlib.pyplot для построения графиков
from nltk.corpus import stopwords  # Импорт списка стоп-слов из библиотеки nltk
from sklearn.feature_extraction.text import TfidfVectorizer  # Импорт класса TfidfVectorizer для векторизации текста
from sklearn.preprocessing import StandardScaler           # Импорт класса StandardScaler для стандартизации данных
from sklearn.model_selection import train_test_split       # Импорт функции train_test_split для разделения данных на обучающую и тестовую выборки
from sklearn.svm import SVC         # Импорт класса SVC для реализации метода опорных векторов
from sklearn.preprocessing import OneHotEncoder  # Импорт класса OneHotEncoder для кодирования категориальных признаков
from sklearn.impute import SimpleImputer  # Импорт класса SimpleImputer из модуля sklearn.impute
from sklearn.metrics import f1_score      # Импорт функции f1_score из модуля sklearn.metrics


"""## **Первичный анализ данных**"""

df = pd.read_csv('C:/Users/mhl_p/source/repos/Digital_breakthrough/train.csv')

date_columns = ['Дата восстановления', 'Дата обращения', "Дата закрытия обращения", "Крайний срок"]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

print(f"Количество записей: {df.shape[0]}")  # Подсчет количества записей в датафрейме

service_counts = df['Сервис'].value_counts() # Подсчет количества уникальных значений в столбце "Категория"
print("Количество записей по сервисам:")
print(service_counts)

functional_group_counts = df['Функциональная группа'].value_counts()             # подсчет количества уникальных значений в столбце "Функциональная группа"
print("Количество записей по функциональной группе:")
print(functional_group_counts)

df['Время обработки'] = df['Дата закрытия обращения'] - df['Дата обращения']     # Вычисление среднего значения в столбце "Время обработки"
mean_processing_time = df['Время обработки'].mean()
print(f"Среднее время обработки: {mean_processing_time} дней")

priority_counts = df.groupby('Приоритет').size()                                 # Группировка записей по столбцу "Приоритет" и подсчет количества записей в каждой группе
print("Количество записей по приоритету:")
print(priority_counts)

"""# **ПОДГОТОВКА ДАННЫХ**

## **train.csv**
"""

df.dropna(subset=['Содержание'], inplace=True)                                     # Удаление строк, у которых в столбце 'Содержание' есть пропущенные значения.

date_columns = ['Дата восстановления', 'Дата обращения', "Дата закрытия обращения", "Крайний срок"]
df[date_columns] = df[date_columns].apply(pd.to_datetime)                          # Преобразование столбцов с датами в формат datetime

mean_diff = (df['Дата восстановления'] - df['Дата обращения']).mean()              # Расчет средней разницы между столбцами 'Дата восстановления' и 'Дата обращения'
df['Дата восстановления'].fillna(df['Дата обращения'] + mean_diff, inplace=True)   # заполнение пропущенных значений в 'Дата восстановления' средним значением
df['Дата восстановления'] = df['Дата восстановления'].dt.date                      # преобразование столбца 'Дата восстановления' в формат date



mean_diff2 = (df['Дата закрытия обращения'] - df['Дата обращения']).mean()         # Расчет средней разницы между столбцами 'Дата закрытия обращения' и 'Дата обращения'
df['Дата закрытия обращения'].fillna(df['Дата обращения'] + mean_diff2, inplace=True)  # заполнение пропущенных значений в 'Дата закрытия обращения' средним значением
df['Дата закрытия обращения'] = df['Дата закрытия обращения'].dt.date              # преобразование столбца 'Дата закрытия обращения' в формат date

df['Приоритет'] = df['Приоритет'].str[0].astype(int)                             # Извлекает первый символ из столбца 'Приоритет', преобразует его в целое число и присваивает обратно в столбец 'Приоритет'
df['Критичность'] = df['Критичность'].str[0].astype(int)                         # Извлекает первый символ из столбца 'Критичность', преобразует его в целое число и присваивает обратно в столбец 'Критичность'
df['Влияние'] = df['Влияние'].str[0].astype(int)                                 # Извлекает первый символ из столбца 'Влияние', преобразует его в целое число и присваивает обратно в столбец 'Влияние'
df['Место'] = df['Место'].str.replace('[^\d.]+', '', regex=True).replace({"": 0}).astype(int)    # Удаляет все символы, кроме цифр и точки, из столбца 'Место', заменяет пустые строки на 0 и преобразует значения в целые числа
df['Сервис'] = df['Сервис'].str.replace('[^\d.]+', '', regex=True).replace({"": 0}).astype(int)  # Удаляет все символы, кроме цифр и точки, из столбца 'Сервис', заменяет пустые строки на 0 и преобразует значения в целые числа
df['Статус'] = df['Статус'].map({"Закрыт": 0, 'Отменен': 1})                                     # Заменяет значения в столбце 'Статус' с помощью словаря: 'Закрыт' на 0, 'Отменен' на 1
df['Система'] = df['Система'].str.replace('[^\d.]+', '', regex=True).replace({"": 0}).astype(int)             # Удаляет все символы, кроме цифр и точки, из столбца 'Система', заменяет пустые строки на 0 и преобразует значения в целые числа
df['Функциональная группа'] = df['Функциональная группа'].str.replace('[^\d.]+', '', regex=True).astype(int)  # Удаляет все символы, кроме цифр и точки, из столбца 'Функциональная группа' и преобразует значения в целые числа
df['Тип обращения итоговый'] = df['Тип обращения итоговый'].replace({"Запрос": 0, 'Инцидент': 1})             # Заменяет значения в столбце 'Тип обращения итоговый' с помощью словаря: 'Запрос' на 0, 'Инцидент' на 1
df['Тип обращения на момент подачи'] = df['Тип обращения на момент подачи'].replace({"Запрос": 0, 'Инцидент': 1})  # Заменяет значения в столбце 'Тип обращения на момент подачи' с помощью словаря: 'Запрос' на 0, 'Инцидент' на 1

scaler = StandardScaler()  # Создает экземпляр класса StandardScaler, который будет использоваться для стандартизации данных

columns_to_standardize = ['Сервис','Приоритет','Функциональная группа','Критичность','Влияние','Система','Место']  # Создает список столбцов, которые требуется стандартизировать

df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize]).round(3)  # Стандартизирует выбранные столбцы данных с помощью метода fit_transform() объекта StandardScaler 
#  Значения в столбцах будут заменены на их стандартизованные значения, округленные до 3 десятичных знаков.

"""## **test.csv**"""

df_test = pd.read_csv("C:/Users/mhl_p/source/repos/Digital_breakthrough/test.csv")

df_test.dropna(subset=['Содержание'], inplace=True)                                                 # Удаляет строки, в которых значение столбца 'Содержание' отсутствует (NaN).

date_columns = ['Дата восстановления', 'Дата обращения', "Дата закрытия обращения", "Крайний срок"]
df_test[date_columns] = df[date_columns].apply(pd.to_datetime)                                      # Преобразует значения в выбранных столбцах в формат даты и времени с помощью функции pd.to_datetime()

mean_diff = (df_test['Дата восстановления'] - df_test['Дата обращения']).mean()                     # Вычисляет среднюю разницу между столбцами 'Дата восстановления' и 'Дата обращения'

df_test['Дата восстановления'].fillna(df_test['Дата обращения'] + mean_diff, inplace=True)          # Заполняет пропущенные значения в столбце 'Дата восстановления' средним значением разницы между 'Дата восстановления' и 'Дата обращения'
# Изменяет значения на месте (inplace=True)

df_test['Дата восстановления'] = df_test['Дата восстановления'].dt.date                             # Преобразует значения в столбце 'Дата восстановления' в формат даты

mean_diff2 = (df_test['Дата закрытия обращения'] - df['Дата обращения']).mean()                     # Вычисляет среднюю разницу между столбцами 'Дата закрытия обращения' и 'Дата обращения'

df_test['Дата закрытия обращения'].fillna(df_test['Дата обращения'] + mean_diff2, inplace=True)     # Заполняет пропущенные значения в столбце 'Дата закрытия обращения' средним значением разницы между 'Дата закрытия обращения' и 'Дата обращения'
# Изменяет значения на месте (inplace=True)

df_test['Дата закрытия обращения'] = df_test['Дата закрытия обращения'].dt.date                     # Преобразует значения в столбце 'Дата закрытия обращения' в формат даты

df_test['Приоритет'] = df_test['Приоритет'].str.slice(0, 1).astype(float)       # Оставляет только первый символ в столбце 'Приоритет' и преобразует его в тип float
df_test['Критичность'] = df_test['Критичность'].str.slice(0, 1).astype(float)   # Оставляет только первый символ в столбце 'Критичность' и преобразует его в тип float
df_test['Влияние'] = df_test['Влияние'].str.slice(0, 1).astype(float)           # Оставляет только первый символ в столбце 'Влияние' и преобразует его в тип float
df_test['Статус'] = df_test['Статус'].replace({"Закрыт": 0, 'Отменен': 1})      # Заменяет значения в столбце 'Статус' с помощью словаря замены
df_test['Сервис'] = df_test['Сервис'].str.replace('[^\d.]+', '', regex=True)    # Удаляет все символы, кроме цифр и точек, в столбце 'Сервис'
df_test['Сервис'] = df_test['Сервис'].replace({"": 0}).astype(int)              # Заменяет пустые строки в столбце 'Сервис' нулем и преобразует в тип int
df_test['Место'] = df_test['Место'].str.replace('[^\d.]+', '', regex=True)      # Удаляет все символы, кроме цифр и точек, в столбце 'Место'
df_test['Место'] = df_test['Место'].replace({"": 0}).astype(int)                # Заменяет пустые строки в столбце 'Место' нулем и преобразует в тип int
df_test['Система'] = df_test['Система'].str.replace('[^\d.]+', '', regex=True)  # Удаляет все символы, кроме цифр и точек, в столбце 'Система'
df_test['Система'] = df_test['Система'].replace({"": 0}).astype(int)            # Заменяет пустые строки в столбце 'Система' нулем и преобразует в тип int
df_test['Функциональная группа'] = df_test['Функциональная группа'].str.replace('[^\d.]+', '', regex=True).astype(int)        # Удаляет все символы, кроме цифр и точек, в столбце 'Функциональная группа' и преобразует его в тип int
df_test['Тип обращения на момент подачи'] = df_test['Тип обращения на момент подачи'].replace({"Запрос": 0, 'Инцидент': 1})   # Заменяет значения в столбце 'Тип обращения на момент подачи' с помощью словаря замены

scaler = StandardScaler()  # Создание экземпляра объекта StandardScaler для стандартизации данных.
columns_to_standardize = ['Сервис', 'Приоритет', 'Функциональная группа', 'Критичность', 'Влияние', 'Система', 'Место']
df_test[columns_to_standardize] = scaler.fit_transform(df_test[columns_to_standardize]).round(3)  # Здесь мы создаем объект StandardScaler для стандартизации данных

"""## **Лемматизация и векторизация признаков train.csv + test.csv**

**Для столбца содержание**
"""

nltk.download('stopwords')  # Загрузка списка стоп-слов из библиотеки NLTK
stop_words = set(stopwords.words('russian'))  # Создание множества стоп-слов на русском языке

df['Содержание'] = df['Содержание'].apply(lambda x: re.sub('[^а-яА-Я]', ' ', str(x)))         # Удаление символов, отличных от русских букв
df['Содержание'] = df['Содержание'].apply(lambda x: x.lower())                                # Приведение к нижнему регистру

df_test['Содержание'] = df_test['Содержание'].apply(lambda x: re.sub('[^а-яА-Я]', ' ', str(x)))  # Удаление символов, отличных от русских букв
df_test['Содержание'] = df_test['Содержание'].apply(lambda x: x.lower())                      # Приведение к нижнему регистру

morph_sod_test = pymorphy2.MorphAnalyzer()  # Создание экземпляра морфологического анализатора для обработки содержания в тестовом наборе данных
df_test['Содержание'] = df_test['Содержание'].apply(lambda x: ' '.join([morph_sod_test.parse(word)[0].normal_form for word in x.split() if word not in stop_words]))  # Применение морфологического анализа и удаление стоп-слов в тестовом наборе данных

vectorizer_sod_test = TfidfVectorizer()     # Создание экземпляра векторизатора TF-IDF для содержания в тестовом наборе данных
X_sod_test = vectorizer_sod_test.fit_transform(df_test['Содержание'])                         # Применение векторизатора TF-IDF к содержанию в тестовом наборе данных

feature_names_sod_test = vectorizer_sod_test.get_feature_names_out()                          # Получение списка имен признаков из векторизатора TF-IDF в тестовом наборе данных
df_features_sod_test = pd.DataFrame(X_sod_test.toarray(), columns=feature_names_sod_test)     # Создание датафрейма признаков на основе векторов TF-IDF в тестовом наборе данных

morph_sod = pymorphy2.MorphAnalyzer()  # Создание экземпляра морфологического анализатора для обработки содержания
df['Содержание'] = df['Содержание'].apply(lambda x: ' '.join([morph_sod.parse(word)[0].normal_form for word in x.split() if word not in stop_words]))  # Применение морфологического анализа и удаление стоп-слов

vectorizer_sod = TfidfVectorizer()                  # Создание экземпляра векторизатора TF-IDF для содержания
X = vectorizer_sod.fit_transform(df['Содержание'])  # Применение векторизатора TF-IDF к содержанию

feature_names = vectorizer_sod.get_feature_names_out()          # Получение списка имен признаков из векторизатора TF-IDF
df_features = pd.DataFrame(X.toarray(), columns=feature_names)  # Создание датафрейма признаков на основе векторов TF-IDF

df_features_sod = pd.concat([df_features_sod_test, df_features], axis=1)  # Объединение датафреймов признаков из тестового и обучающего наборов данных

"""**Для столбца Решение**"""

df.iloc[:, 9] = df.iloc[:, 9].apply(lambda x: re.sub('[^а-яА-Я]', ' ', str(x)))  # Удаление символов, отличных от русских букв
df.iloc[:, 9] = df.iloc[:, 9].apply(lambda x: x.lower())                         # Приведение к нижнему регистру

df_test.iloc[:, 10] = df_test.iloc[:, 10].apply(lambda x: re.sub('[^а-яА-Я]', ' ', str(x)))  # Удаление символов, отличных от русских букв
df_test.iloc[:, 10] = df_test.iloc[:, 10].apply(lambda x: x.lower())                         # Приведение к нижнему регистру

morph_res_test = pymorphy2.MorphAnalyzer()  # Создание экземпляра морфологического анализатора для обработки результата в тестовом наборе данных
df_test.iloc[:, 10] = df_test.iloc[:, 10].apply(lambda x: ' '.join([morph_res_test.parse(word)[0].normal_form for word in x.split() if word not in stop_words]))  # Применение морфологического анализа и удаление стоп-слов в тестовом наборе данных для результата

vectorizer_res_test = TfidfVectorizer()                               # Создание экземпляра векторизатора TF-IDF для результата в тестовом наборе данных
X_res_test = vectorizer_res_test.fit_transform(df_test.iloc[:, 10])   # Применение векторизатора TF-IDF к результату в тестовом наборе данных

feature_names_res_test = vectorizer_res_test.get_feature_names_out()  # Получение списка имен признаков из векторизатора TF-IDF в тестовом наборе данных для результата
df_features_res_test = pd.DataFrame(X_res_test.toarray(), columns=feature_names_res_test)  # Создание датафрейма признаков на основе векторов TF-IDF в тестовом наборе данных для результата

morph = pymorphy2.MorphAnalyzer()                                     # Создание экземпляра морфологического анализатора для обработки результата
df.iloc[:, 9] = df.iloc[:, 9].apply(lambda x: ' '.join([morph.parse(word)[0].normal_form for word in x.split() if word not in stop_words]))  # Применение морфологического анализа и удаление стоп-слов для результата

vectorizer = TfidfVectorizer()                                        # Создание экземпляра векторизатора TF-IDF для результата
X_res = vectorizer.fit_transform(df.iloc[:, 9])                       # Применение векторизатора TF-IDF к результату

feature_names_res = vectorizer.get_feature_names_out()                # Получение списка имен признаков из векторизатора TF-IDF для результата
df_features_res = pd.DataFrame(X_res.toarray(), columns=feature_names_res)    # Создание датафрейма признаков на основе векторов TF-IDF для результата

df_features_res = pd.concat([df_features_res_test, df_features_res], axis=1)  # Объединение датафреймов признаков для результата

df.head()

df_test.head()

"""## **0 Обучение модели с признаками Содержание и Решение**"""

selected_features = ['Сервис', 'Приоритет', 'Статус', 'Функциональная группа', 'Тип обращения на момент подачи', 'Критичность', 'Влияние', 'Система', 'Место']
df_selected_features = df[selected_features].reset_index(drop=True)                     # Выбор и сброс индексов выбранных признаков в датафрейме df

df_train = pd.concat([df_selected_features, df_features_sod, df_features_res], axis=1)  # Объединение датафреймов df_selected_features, df_features_sod и df_features_res по столбцам

X_train = df_train.values                     # Преобразование датафрейма df_train в массив значений для признаков (обучающая выборка)
y_train = df['Тип переклассификации'].values  # Присвоение массиву значений целевой переменной 'Тип переклассификации' из датафрейма df

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  # Разделение обучающей выборки на обучающий и тестовый наборы

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

imputer = SimpleImputer(strategy='mean')         # Создание объекта SimpleImputer

X_train_filled = imputer.fit_transform(X_train)  # Применение imputer к X_train

model = SVC()                                    # Обучение модели на заполненных данных
model.fit(X_train_filled, y_train)

X_test_filled = imputer.transform(X_test)  # Применение imputer к X_test

y_pred = model.predict(X_test_filled)      # Предсказание на заполненных данных

accuracy = (y_pred == y_test).mean()       # Вычисление точности
print(f"Точность: {accuracy}")

f1_macro = f1_score(y_test, y_pred, average='macro')  # Вычисление макро-усредненной F1-меры
print("Macro-average F1-score:", f1_macro)            # Вывод значения макро-усредненной F1-меры

"""## **1 Обучение модели без признаков Содержание и Решение**"""

X_train = df[['Сервис', 'Приоритет', 'Статус', 'Функциональная группа', 'Тип обращения на момент подачи', 'Критичность',	'Влияние',	'Система',	'Место']]

y_train = df['Тип переклассификации']

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = (y_pred == y_test).mean()
print(f"Точность: {accuracy}")

f1_macro = f1_score(y_test, y_pred, average='macro')  # Вычисление макро-усредненной F1-меры
print("Macro-average F1-score:", f1_macro)            # Вывод значения макро-усредненной F1-меры

X_train = df[['Сервис', 'Приоритет', 'Статус', 'Функциональная группа', 'Тип обращения на момент подачи', 'Критичность',	'Влияние',	'Система',	'Место']]
y_train = df['Тип обращения итоговый']

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model2 = SVC()
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

accuracy = (y_pred == y_test).mean()
print(f"Точность: {accuracy}")

f1_macro = f1_score(y_test, y_pred, average='macro')  # Вычисление макро-усредненной F1-меры
print("Macro-average F1-score:", f1_macro)            # Вывод значения макро-усредненной F1-меры

"""## **Тестовый датасет test.csv**"""

df_test.head()

X_test = df_test[['Сервис', 'Приоритет', 'Статус', 'Функциональная группа', 'Тип обращения на момент подачи', 'Критичность',	'Влияние',	'Система',	'Место']]

y_pred_pereclass = model.predict(X_test)           # предсказания Тип переклассификации

y_pred_itog = model2.predict(X_test)               # предсказания Тип обращения итоговый

print(len(y_pred_pereclass[y_pred_pereclass!=0]))  # количество ненулевых данных

print(len(y_pred_itog[y_pred_itog!=0]))            # количество ненулевых данных

"""## Количество переклассификаций в тестовом датасете test.csv"""

count_nonzero = np.count_nonzero(y_pred)

print(f"Количество переклассификаций: {count_nonzero}")

"""## **Вывод результатов в файл**"""

submission_df = pd.read_csv('/content/drive/MyDrive/hacaton/submission.csv')  # Загрузка существующего файла submission.csv

submission_df['Тип переклассификации'] = y_pred_pereclass                            # Заполнение столбцов "Тип переклассификаций"
submission_df['Тип обращения итоговый'] = y_pred_itog                                # Заполнение столбцов "Тип обращения итоговый"

submission_df.to_csv('/content/drive/MyDrive/hacaton/submission.csv', index=False)   # Сохранение обновленного DataFrame в файл CSV