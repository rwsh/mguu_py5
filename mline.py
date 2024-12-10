import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Считывание данных из Excel
file_name = "mdata.xlsx"  # Имя файла
data = pd.read_excel(file_name)

# Проверка данных
print("Данные из файла:")
print(data.head())

# Разделение данных на независимые (X) и зависимую (Y) переменные
X = data[['X1', 'X2']]  # Независимые переменные
Y = data['Y']           # Зависимая переменная

# Создание модели линейной регрессии
model = LinearRegression()
model.fit(X, Y)

# Вывод коэффициентов
print("\nКоэффициенты множественной линейной регрессии:")
print(f"Угловые коэффициенты (slopes): {model.coef_}")
print(f"Свободный член (intercept): {model.intercept_}")

# Прогнозирование
Y_pred = model.predict(X)

# Расчет коэффициента детерминации R^2
r2 = r2_score(Y, Y_pred)
print(f"\nКоэффициент детерминации (R^2): {r2:.4f}")

# Визуализация результатов (покажем реальное и предсказанное значение)
plt.figure(figsize=(8, 5))
plt.plot(Y, label='Реальные значения', color='blue', marker='o')
plt.plot(Y_pred, label='Предсказанные значения', color='red', linestyle='--', marker='x')
plt.title('Реальные и предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Y')
plt.legend()
plt.show()
