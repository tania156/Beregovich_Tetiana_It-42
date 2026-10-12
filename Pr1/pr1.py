import numpy as np

temps = np.array([-4, -3, 2, 9, 15, 19, 21, 20, 14, 8, 2, -2])
days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
month_numbers = np.arange(1, 13)

print("temps shape:", temps.shape, "dtype:", temps.dtype)
print("days_in_month shape:", days_in_month.shape, "dtype:", days_in_month.dtype)
print("month_numbers shape:", month_numbers.shape, "dtype:", month_numbers.dtype)

temps_f = temps * 9 / 5 + 32
warm_days_total = np.sum(days_in_month[temps > 15])
is_365 = days_in_month.sum() == 365

print("temps_f:", temps_f)
print("warm_days_total:", warm_days_total)
print("is_365:", is_365)

warmest_val = temps.max()
warmest_idx = temps.argmax()
coldest_val = temps.min()
coldest_idx = temps.argmin()

print("Warmest:", warmest_val, "at index", warmest_idx)
print("Coldest:", coldest_val, "at index", coldest_idx)

months_below_zero = month_numbers[temps < 0]
print("Months below zero:", months_below_zero)

sorted_temps = np.sort(temps)
print("Sorted temps:", sorted_temps)
print("Original temps:", temps)

sunshine_hours = np.array([35, 52, 107, 169, 275, 298, 301, 280, 197, 115, 50, 31])
data_2d = np.array([temps, days_in_month, sunshine_hours])
print("2D shape:", data_2d.shape)

row_sums = data_2d.sum(axis=1)
row_means = data_2d.mean(axis=1)
print("Row sums:", row_sums)
print("Row means:", row_means)

col_sums = data_2d.sum(axis=0)
print("Col sums:", col_sums)

rows = np.array([temps, days_in_month])
correction = np.array([1, 0])

try:
    result_err = rows + correction
except Exception as e:
    print("Error:", e)

correction_reshaped = correction.reshape(2, 1)
rows_corrected = rows + correction_reshaped
print("Corrected rows:\n", rows_corrected)

answers = """
1. shape == (12,) це одновимірний масив із 12 елементів, а shape == (1, 12) це двовимірний масив з 1 рядком і 12 стовпцями.
2. temps[temps > 15] повертає масив іншої довжини, оскільки булева індексація фільтрує елементи за умовою, залишаючи лише ті місяці, де температура вища за 15°C.
3. Додавання масиву форми (3,) до масиву форми (3, 12) викличе ValueError через несумісність вимірів для broadcasting.
4. rows + correction дає помилку, бо форми (2, 12) і (2,) несумісні. Після reshape(2, 1) форма стає (2, 1), і вимір розміром 1 успішно розширюється уздовж 12 стовпців.
"""
print(answers)