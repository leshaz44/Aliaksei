# Подсчет пустых значений в каждом столбце
import pandas as pd

def count_missing_values(df):
    """Подсчитывает количество пустых значений в каждом столбце."""
    return df.isnull().sum()