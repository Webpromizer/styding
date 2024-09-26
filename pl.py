import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import requests
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


# df = pd.DataFrame({
#     'Имя': ['Иван', 'Мария', 'Алексей'],
#     'Возраст': [28, 34, 18],
#     'Город': ['Донецк', 'Запорожье', 'Горжице']
# })
# df['Фамилия'] = ['Пушкин', 'Печкин', 'Пупкин']

# df_older_25 = df[df['Возраст'] > 25]
# print(df_older_25)
# average_age = df['Возраст'].mean()
# print(round(average_age))

# fig, ax = plt.subplots(figsize=(8, 3)) 
# ax.axis('tight')
# ax.axis('off')
# table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
# plt.show()

# s = pd.Series([1,3,5,7], index=['a', 'b', 'c', 'd'])
# s.index = [10, 35, 21, 54]
# s_sum = s.sum()
# print(s_sum)
# print(s)
# # Создание фигуры и осей
# fig, ax = plt.subplots(figsize=(5, 2))  # Размер окна можно настроить
# ax.axis('tight')
# ax.axis('off')
# # Создание таблицы
# # Данные для таблицы
# cell_text = [[str(val)] for val in s]
# row_labels = s.index
# # Добавление таблицы
# table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=['Значения'], loc='center')
# # Отображение таблицы
# plt.show()
# dat_frame = {'Name': ['Ivan', 'Pasha', 'Kolya', 'Evgen', 'Zlata'],
#              'Last_name': ['Pupkin','Solya','Gavkin','Myaukin','Supkin'],
#              'Work': ['Teh', 'Dev', 'ingeneer', 'advokat', 'bomzh']
# }
# df = pd.DataFrame(dat_frame)
# df.to_csv('dat.csv', index=False)

# df = pd.read_csv('dat.csv', sep=';', header=0)

# df = pd.DataFrame(np.random.randint(low = 0, high=100, size=(5, 2)), index=['A', 'B', 'C', 'D', 'E'], columns=['Value1', 'Value2'])

# value2 = df['Value2']

# value_of_C = value2['C']
# print("Value of 'Value2' for index label 'C':", value_of_C)

# value_of_third_row = value2.iloc[2]
# print("Value of 'Value2' for the third row:", value_of_third_row)

# df = pd.DataFrame(np.random.randint(low=0, high=50, size=(7, 3)), index=['Mondey', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday', 'Saturday'], columns=['precipitation', 'humidity','temp'] )
# fig, ax = plt.subplots(figsize=(7, 3)) 
# ax.axis('tight')
# ax.axis('off')
# table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
# plt.show()
# humidity2 = df['humidity']
# humidity_of_tu = humidity2['Tuesday']
# print("Humidity in Tuesday: ", humidity_of_tu)

# create_In = pd.Index(['A', 'B', 'C', 'D', 'E'])
# df = pd.Series(np.arange(1, 6, 1), index=(create_In))
# rename_In = {'A': 'Index_A', 'B': 'Index_B', 'C': 'Index_C', 'D': 'Index_D', 'E': 'Index_E'}
# new_In = df.rename(index=rename_In)
# print(new_In)

# S = pd.Series([1, None, 45, None], index=([1, 3, 6, 5]))
# fill = S.fillna(6, inplace=True)
# print(S)

# multi = pd.MultiIndex.from_product([['Q1', 'Q2', 'Q3', 'Q4'], ['summer', 'winter', 'spring', 'autumn']])
# df = pd.DataFrame(np.arange(16), index=(multi))
# multi.set_names(['Quarter', 'Season'], inplace=True)
# months = ['November', 'October', 'December', 'January']
# new_levels = [multi.levels[0], months]
# new_multi = multi.set_levels([multi.levels[0], months])
# df.index = new_multi
# print(df.index)


# АНАЛИЗ ПОГОДЫ

# api_key = "b4cae65c3f850a892d118b388d7fdf24"
# city = "Hradec Králové"
# url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=ru"

# response = requests.get(url)
# data = response.json()

# if response.status_code == 200:
#     weather_data = {
#         "city": city,
#         "temperature": data["main"]["temp"],
#         "pressure": data["main"]["pressure"],
#         "humidity": data["main"]["humidity"],
#         "weather": data["weather"][0]["description"],
#         "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
#     }
# else:
#     print(f"Ошибка получения данных: {data['message']}")

# df = pd.DataFrame([weather_data])
# df.to_csv("weather_data.csv", index=False)

# df = pd.read_csv('weather_data.csv')
# df['date'] = pd.to_datetime(df['date'])  

# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# df = df.dropna()

# temperatures = df['temperature'].values 
# print(temperatures)
# mean_temp = np.mean(temperatures)
# print(f"Средняя температура: {mean_temp}")
# median_temp = np.median(temperatures)
# print(f"Медианная температура: {median_temp}")
# std_dev_temp = np.std(temperatures)
# print(f"Стандартное отклонение температуры: {std_dev_temp}")

# plt.plot(df['date'], df['temperature'])
# plt.title('Temperature Over Time')
# plt.xlabel('Date')
# plt.ylabel('Temperature (°C)')
# plt.show()

# plt.hist(df['humidity'], bins=30)
# plt.title('Humidity Distribution')
# plt.xlabel('Humidity')
# plt.ylabel('Frequency')
# plt.show()

# df.boxplot(column=['temperature', 'humidity'], by='date')
# plt.title('Monthly Climate Statistics')
# plt.show()


# Анализ csv файла с фильмами

# file_path = 'tmdb_5000_credits.csv'
# read = pd.read_csv(file_path)

# print(read.columns)

# # print(read.describe()) 
# # print(read.info())
# # print(read.isnull()) - есть ли пропуски

# plt.figure(figsize = (10, 6))
# read['movie_id'].value_counts().plot(kind="bar")
# plt.title('Distribution of Movies by movie_id')
# plt.xlabel('movie_id')  # Подпись оси X
# plt.ylabel('Number of Movies')  # Подпись оси Y
# plt.show()


# plt.figure(figsize = (10, 6))
# read['title'].value_counts().sort_index().plot(kind="bar")
# plt.title('Distribution of Movies by title')
# plt.xlabel('title')  # Подпись оси X
# plt.ylabel('Number of Movies')  # Подпись оси Y
# plt.show()

# # Подсчет упоминаний в колонке 'cast'
# cast_mentions = read('cast')

# # Выбор топ-30 самых частых упоминаний
# top_cast = cast_mentions.most_common(30)
# cast_names, cast_counts = zip(*top_cast)

# # Построение бар-графика для 'cast'
# plt.figure(figsize=(10, 6))
# plt.bar(cast_names, cast_counts, edgecolor='black')
# plt.title('Top 30 Most Common Cast Members')
# plt.xlabel('Cast Member')
# plt.ylabel('Frequency')
# plt.xticks(rotation=90)  # Поворот подписей оси X для лучшей читаемости
# plt.show()

# # Взаимосвязь между рейтингом фильма и его бюджетом
# plt.figure(figsize=(10, 6))
# plt.scatter(read['budget'], read['rating'], alpha=0.5)
# plt.title('Relationship between Movie Budget and Rating')
# plt.xlabel('Budget (in millions)')
# plt.ylabel('Rating')
# plt.show()

# # Взаимосвязь между рейтингом фильма и его кассовыми сборами
# plt.figure(figsize=(10, 6))
# plt.scatter(read['box_office'], read['rating'], alpha=0.5)
# plt.title('Relationship between Movie Box Office and Rating')
# plt.xlabel('Box Office (in millions)')
# plt.ylabel('Rating')
# plt.show()



# tips = sns.load_dataset('tips')
# print(tips.head())

# sns.histplot(tips['total_bill'], kde=True)
# plt.title('Distribution of Total Bill')
# plt.show()

# sns.scatterplot(x='total_bill', y='tip', data=tips)
# plt.title('Total Bill vs Tip')
# plt.show()

# sns.scatterplot(x='total_bill', y='tip', hue="time", data=tips)
# plt.title('Total Bill vs Tip')
# plt.show()

# sns.scatterplot(x='total_bill', y='tip', size='size', hue='time', data=tips)
# plt.title('Total Bill vs Tip')
# plt.show()

# sns.scatterplot(x='total_bill', y='tip', hue='time', style="sex", data=tips)
# plt.title('Total Bill vs Tip')
# plt.show()

# sns.regplot(x='total_bill', y='tip', data=tips)
# plt.title('Total Bill vs Tip with Regression Line')
# plt.show()

# g = sns.FacetGrid(tips, col='time', row="sex")
# g.map(sns.scatterplot, 'total_bill', 'tip')
# plt.suptitle('FacetGrid of Total Bill vs Tip by Time and Sex', y=1.02)
# plt.show()

# sns.set_style('whitegrid')
# sns.set_palette('muted')
# sns.set_context('talk')

# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='total_bill', y='tip', hue='day', style='time', size='size', data=tips, sizes=(20, 200))
# plt.title('Total Bill vs Tip by Day, Time and Size')
# plt.show()
   
# plt.figure(figsize=(18, 12))

# # Гистограмма
# plt.subplot(2, 3, 1)
# sns.histplot(tips['total_bill'], bins=20, kde=False, color="y")
# plt.title('Histogram of Total Bill')
# plt.xlabel('Total Bill')
# plt.ylabel('Frequency')

# # KDE-график
# plt.subplot(2, 3, 2)
# sns.kdeplot(tips['total_bill'], shade=True, color="g")
# plt.title('KDE Plot of Total Bill')
# plt.xlabel('Total Bill')
# plt.ylabel('Density')

# # Комбинированная гистограмма и KDE-график
# plt.subplot(2, 3, 3)
# sns.histplot(tips['total_bill'], bins=20, kde=True, color="b")
# plt.title('Histogram and KDE Plot of Total Bill')
# plt.xlabel('Total Bill')
# plt.ylabel('Frequency/Density')

# # Boxplot
# plt.subplot(2, 3, 4)
# sns.boxplot(x='day', y='total_bill', data=tips, color="r")
# plt.title('Boxplot of Total Bill by Day')
# plt.xlabel('Day')
# plt.ylabel('Total Bill')

# # Violin Plot
# plt.subplot(2, 3, 5)
# sns.violinplot(x='day', y='total_bill', data=tips, color="orange")
# plt.title('Violin Plot of Total Bill by Day')
# plt.xlabel('Day')
# plt.ylabel('Total Bill')

# # Добавим пустой график для симметрии
# plt.subplot(2, 3, 6)
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.barplot(y="total_bill", x="day", data=tips )
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.countplot(x='day', data=tips)
# plt.title('Count Plot of Days')
# plt.xlabel('Day')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.pointplot(x='day', y='total_bill', data=tips)
# plt.title('Point Plot of Total Bill by Day')
# plt.xlabel('Day')
# plt.ylabel('Average Total Bill')
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.swarmplot(x='day', y='total_bill', data=tips)
# plt.title('Swarm Plot of Total Bill by Day')
# plt.xlabel('Day')
# plt.ylabel('Total Bill')
# plt.show()

dates = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(4)]
product_arr = ["Bread","Milk","Water","Guns"]
sales_arr = np.random.randint(1, 11, size=4)
revenue_arr = np.floor(np.random.randint(10, 50, size=4))

df = pd.DataFrame({
    "Data": dates,
    "Product": product_arr,
    "Sales": sales_arr,
    "Revenue": revenue_arr
})

df.to_csv("practice", index=False)
read_df = pd.read_csv("practice")

read_df.info()

# Вычисление среднего дохода на единицу проданного товара
df['Revenue_per_Unit'] = np.floor(df['Revenue'] / df['Sales'])

# Построение графика изменения выручки по датам (Matplotlib)
fix, ax = plt.subplots()
ax.plot(df["Data"], df["Sales"])

ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.set_title('Sales Over Time')

# plt.show()

# Построение графика распределения количества продаж для разных продуктов (Seaborn)
plt.figure(figsize=(10, 6))
sns.barplot(x='Product', y='Sales', data=df, palette='viridis')

plt.title('Sales Distribution by Product')
plt.xlabel('Product')
plt.ylabel('Number of Sales')
plt.xticks(rotation=45)

# plt.show()


# Парные графики (pairplot) для всех числовых столбцов.
sns.pairplot(df)
plt.suptitle('Pairwise Plots of Product Data by Category', y=1.02)

# plt.show()

# График зависимости выручки от количества продаж, с цветом для обозначения разных продуктов 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Sales', y='Revenue', hue='Product', palette='viridis')

plt.xlabel('Sales')
plt.ylabel('Revenue')
plt.title('Revenue vs Sales by Product')

plt.show()

df.to_csv("complete_practice!", index=False)

import tensorflow as tf

Отключение oneDNN сообщений
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Отключение oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



#region создание неизменяемого тензора 
a = tf.constant([[1, 2],
             [2, 4],
             [5, 8]], dtype=tf.float16)
b = tf.constant([1, 3, 5, 7])
c = tf.constant(1, shape=(1, 1))
# print(a) -------------------------------
# endregion

#region создание переменной(изменяемого тензора)
v1 = tf.Variable(-1.4)
v2 = tf.Variable([1, 8, 2, 4], dtype= tf.float32)
v3 = tf.Variable(b)
# print(v1, v2, v3) ---------------------------
# endregion

#region изменение типа данных float32
a2 = tf.cast(a, dtype=tf.float32)
# print(a2) -----------------------
# endregion

#region преобразование тензора в массив numpy
b2 = b.numpy()
# print(b2) ---------------------------
# endregion

#region изменение значения при помощи assign
v1.assign(4)
# print(v1)---------------------------
# endregion

#region добавление значения к существующим при помощи assign_add
v2.assign_add([1, 2, 4, 8])
# print(v2)-------------------------------------
# endregion

#region удаление значения при помощи assign_sub
v3.assign_sub([0, 2, 6, 2])
# print(v3)---------------------------------
# endregion

#region обращен ие к элементам по индексу
x_geth = tf.gather(v2, [0, 3])
# print(x_geth)---------------------
# endregion

#region изменение размерности Тензора 


a_shape = tf.Variable(range(30))
a_resh = tf.reshape(a_shape, [5, 6])
# print(a_resh)--------------------------
# endregion

#region вычисление значения производной

x = tf.Variable(-2.0)

with tf.GradientTape() as tape:
    y = x ** 2

df = tape.gradient(y, x)
# print(df) ------------------------------
# endregion

#region нахождение производного
w = tf.Variable(tf.random.normal(((3, 2))))
b = tf.Variable(tf.zeros(2, dtype=tf.float32))
x = tf.Variable([[-2.0, 1.0, 3.0]])

with tf.GradientTape() as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y ** 2)
df = tape.gradient(loss, [w, b])
# print(df[0], df[1], sep="\n")---------
# endregion

#region Task 1
x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    y = 3*x ** 2 + 2*x + 1

grade = tape.gradient(y, x)
# print(grade)
# endregion

#region Task 2
x = tf.Variable(np.pi/2)

with tf.GradientTape() as tape:
    y = tf.sin(x) + x ** 2

grade_2 = tape.gradient(y, x)
# print(grade_2)
# endregion

#region Task 3
x = tf.Variable(1.0)
e = tf.Variable(np.e)

with tf.GradientTape() as tape:
    y = e * x

grade_3 = tape.gradient(y, x)
# print(grade_3)
# endregion

#region Task 4

x = tf.Variable(1.0)
y = tf.Variable(1.0)

with tf.GradientTape() as tape:
    f = x ** 2 + y ** 2

grade_4 = tape.gradient(f, [x, y])
# print(grade_4)
# endregion

# region Task 5
x = tf.Variable(0.0)

with tf.GradientTape() as tape:
   sigmoida = 1 / (1 + tf.exp(-x))

grade_6 = tape.gradient(sigmoida, x)
# print(grade_6)
# endregion

# region main

TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0,maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = x * k_true + b_true + noise

plt.scatter(x, y, s=2)
# plt.show()

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 500
learning_rate = 0.02



for n in range(EPOCHS):
    with tf.GradientTape() as tape:
        f = k * x + b
        loss = tf.reduce_mean(tf.square(y - f))

    dk, db = tape.gradient(loss, [k, b])

    k.assign_sub(learning_rate * dk)
    b.assign_sub(learning_rate * db)

print(k, b, sep="\n")

y_pr = k * x+b
plt.scatter(x, y, s=2, c="b")
plt.scatter(x, y_pr, c="r", s=2)
# plt.show()
# endregion

# region main_batch
TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0,maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = x * k_true + b_true + noise

plt.scatter(x, y, s=2)
# plt.show()

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 200
learning_rate = 0.02

BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE

for n in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = y[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]

    with tf.GradientTape() as tape:
        f = k * x_batch + b
        loss = tf.reduce_mean(tf.square(y_batch - f))

    dk, db = tape.gradient(loss, [k, b])

    k.assign_sub(learning_rate * dk)
    b.assign_sub(learning_rate * db)

print(k, b, sep="\n")

y_pr = k * x+b
plt.scatter(x, y, s=2, c="b")
plt.scatter(x, y_pr, c="r", s=2)
# plt.show()


# endregion

# region main_SGD
TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0,maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = x * k_true + b_true + noise

plt.scatter(x, y, s=2)
# plt.show()

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 200
learning_rate = 0.02

BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE

opt = tf.optimizers.SGD(learning_rate= 0.02, nesterov = True)

for n in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = y[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]

    with tf.GradientTape() as tape:
        f = k * x_batch + b
        loss = tf.reduce_mean(tf.square(y_batch - f))

    dk, db = tape.gradient(loss, [k, b])

    # k.assign_sub(learning_rate * dk)
    # b.assign_sub(learning_rate * db)
    opt.apply_gradients(zip([dk, db], [k, b]))

print(k, b, sep="\n")

y_pr = k * x+b
plt.scatter(x, y, s=2, c="b")
plt.scatter(x, y_pr, c="r", s=2)
# plt.show()


# endregion

# region main_Adagrad
TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0,maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = x * k_true + b_true + noise

plt.scatter(x, y, s=2)
# plt.show()

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 200
learning_rate = 0.02

BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE

opt = tf.optimizers.Adagrad(learning_rate= 0.06)

for n in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = y[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]

    with tf.GradientTape() as tape:
        f = k * x_batch + b
        loss = tf.reduce_mean(tf.square(y_batch - f))

    dk, db = tape.gradient(loss, [k, b])

    # k.assign_sub(learning_rate * dk)
    # b.assign_sub(learning_rate * db)
    opt.apply_gradients(zip([dk, db], [k, b]))

print(k, b, sep="\n")

y_pr = k * x+b
plt.scatter(x, y, s=2, c="b")
plt.scatter(x, y_pr, c="r", s=2)
# plt.show()


# endregion

# region main_Adadelta

TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0,maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = x * k_true + b_true + noise

plt.scatter(x, y, s=2)
# plt.show()

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 200
learning_rate = 0.02

BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE

opt = tf.optimizers.Adadelta(learning_rate= 5.00)

for n in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = y[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]

    with tf.GradientTape() as tape:
        f = k * x_batch + b
        loss = tf.reduce_mean(tf.square(y_batch - f))

    dk, db = tape.gradient(loss, [k, b])

    # k.assign_sub(learning_rate * dk)
    # b.assign_sub(learning_rate * db)
    opt.apply_gradients(zip([dk, db], [k, b]))

print(k, b, sep="\n")

y_pr = k * x+b
plt.scatter(x, y, s=2, c="b")
plt.scatter(x, y_pr, c="r", s=2)
# plt.show()


# endregion

# region Модель tf.Model

class DenseNN(tf.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)

            self.fl_init = True

        y = x @ self.w + self.b
        return y


model = DenseNN(1)
#print( model(tf.constant([[1.0, 2.0]])) )

x_train = tf.random.uniform(minval=0, maxval=10, shape=(100, 2))
y_train = [a + b for a, b in x_train]

loss = lambda x, y: tf.reduce_mean(tf.square(x - y))
opt = tf.optimizers.Adam(learning_rate=0.01)

EPOCHS = 50
for n in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.expand_dims(x, axis=0)
        y = tf.constant(y, shape=(1, 1))

        with tf.GradientTape() as tape:
            f_loss = loss(y, model(x))

        grads = tape.gradient(f_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    print(f_loss.numpy())

print(model.trainable_variables)

# endregion