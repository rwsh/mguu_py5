import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Считывание данных из Excel
file_name = "data.xlsx"  # Имя файла
data = pd.read_excel(file_name)

# Проверка данных
print("Данные из файла:")
print(data.head())

# Разделение на независимую (X) и зависимую (Y) переменные
X = data[['X']]  # Признак должен быть двумерным для sklearn
Y = data['Y']    # Целевая переменная

# Создание модели линейной регрессии
model = LinearRegression()
model.fit(X, Y)

# Вывод коэффициентов
print("\nКоэффициенты линейной регрессии:")
print(f"Угловой коэффициент (slope): {model.coef_[0]}")
print(f"Свободный член (intercept): {model.intercept_}")

# Прогнозирование
Y_pred = model.predict(X)

# Расчет коэффициента детерминации R^2
r2 = r2_score(Y, Y_pred)
print(f"\nКоэффициент детерминации (R^2): {r2:.4f}")

# Визуализация данных и регрессии
plt.scatter(X, Y, color='blue', label='Данные')
plt.plot(X, Y_pred, color='red', label='Линейная регрессия')
plt.title('Парная линейная регрессия')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
