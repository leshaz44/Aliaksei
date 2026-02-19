#visualization.py
# Визуализация истинных и предсказанных значений
import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, num_points=20):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(num_points), y_true[:num_points], color='blue', label='Истинные значения')
    plt.scatter(range(num_points), y_pred[:num_points], color='red', label='Предсказанные значения')
    plt.xlabel('Индекс')
    plt.ylabel('Значение оплаты труда')
    plt.title(f'Истинные и предсказанные значения оплаты труда (первые {num_points} точек)')
    plt.legend()
    plt.show()

def plot_histogram_errors(y_true, y_pred):
    """
    Функция рисует гистограмму ошибок между true-значениями и predicted-значениями.
    
    :param y_true: Массив настоящих значений (например, зарплат)
    :param y_pred: Массив предсказанных значений
    """
    errors = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
    plt.title("Гистограмма ошибок предиктивной модели")
    plt.xlabel("Ошибка (true - pred)")
    plt.ylabel("Количество наблюдений")
    plt.grid(True)
    plt.show()
def create_scatter_plot(y_true, y_pred, title='Диаграмма рассеяния', xlabel='Ось X', ylabel='Ось Y', color='blue'):
    """
    x_data (list or array): Значения по оси X.
    y_data (list or array): Соответствующие значения по оси Y.
    title (str): Заголовок диаграммы.
    xlabel (str): Подпись оси X.
    ylabel (str): Подпись оси Y.
    color (str): Цвет точек на графике.
    """
    plt.figure(figsize=(10, 6))              # Устанавливаем размер окна графика
    plt.scatter(y_true, y_pred, color=color)  # Рисуем диаграмму рассеяния
    plt.title(title)                          # Добавляем заголовок
    plt.xlabel(xlabel)                       # Подпись оси X
    plt.ylabel(ylabel)                       # Подпись оси Y
    plt.grid(True)                            # Включаем сетку
    plt.show()                                # Показываем график