import numpy as np

print("Практика 1. Варіант 1 — Київ")
print("numpy", np.__version__)

a = np.array([1, 2, 3, 4, 5])
print(a)
print("shape:", a.shape)
print("dtype:", a.dtype)

temps = np.array([-4, -3, 2, 9, 15, 19, 21, 20, 14, 8, 2, -2])
days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
month_numbers = np.arange(1, 13)

print("\nЗавдання 1")
for name, arr in [("temps", temps), ("days_in_month", days_in_month), ("month_numbers", month_numbers)]:
    print(name, arr, "shape:", arr.shape, "dtype:", arr.dtype)

print("\nЗавдання 2")
temps_f = temps * 9 / 5 + 32
print("°F:", temps_f)
warm_days = days_in_month[temps > 15].sum()
print("Днів у році тепліше 15°C:", warm_days)
print("Сума днів:", days_in_month.sum(), days_in_month.sum() == 365)

print("\nЗавдання 3")
print("Найтепліший місяць: індекс", temps.argmax(), "номер", month_numbers[temps.argmax()], "значення", temps.max())
print("Найхолодніший місяць: індекс", temps.argmin(), "номер", month_numbers[temps.argmin()], "значення", temps.min())
print("Місяці з температурою нижче нуля:", month_numbers[temps < 0])
print("Відсортовані температури:", np.sort(temps))
print("Оригінал без змін:", temps)

print("\nЗавдання 4")
sunshine_hours = np.array([40, 65, 130, 185, 260, 275, 290, 265, 185, 115, 45, 30])
data = np.array([temps, days_in_month, sunshine_hours])
print("shape:", data.shape)
print("Сума по рядках:", data.sum(axis=1))
print("Середнє по рядках:", data.mean(axis=1))
col_sums = data.sum(axis=0)
print("Сума по стовпцях:", col_sums)
print(f"Для січня: {temps[0]} + {days_in_month[0]} + {sunshine_hours[0]} = {col_sums[0]} — це сума різнорідних величин (°C, дні, години), тож фізичного змісту число не має; axis=0 лише показує, що агрегація йде по стовпцях.")

print("\nЗавдання 5")
rows = np.array([temps, days_in_month])
correction = np.array([1, 0])
print("rows.shape:", rows.shape, "correction.shape:", correction.shape)
try:
    print(rows + correction)
except ValueError as e:
    print("Помилка:", e)
print("Порівняння з кінця: 12 і 2 — не рівні й жодне не дорівнює 1, тому форми несумісні.")
correction_col = correction.reshape(2, 1)
print("correction_col.shape:", correction_col.shape)
print(rows + correction_col)
print("Тепер (2, 12) і (2, 1): 12 і 1 — сумісні (1 розтягується до 12), 2 і 2 — рівні. Поправка 1 додається до всього першого рядка, 0 — до другого.")

print("""
Відповіді на питання:
1. shape (12,) — одновимірний масив із 12 елементів; shape (1, 12) — двовимірний масив з одного рядка і 12 стовпців. Дані ті самі, але кількість вимірів різна, тому вони по-різному поводяться в індексації та broadcasting.
2. temps > 15 дає булеву маску, а temps[маска] повертає лише елементи, де маска True, тому довжина результату дорівнює кількості таких елементів (у Києві 4: травень—серпень), а не 12.
3. Форми (3,) і (3, 12) порівнюються з кінця: 3 і 12 — несумісні, тому буде ValueError. Щоб додати по одному числу на рядок, треба reshape(3, 1).
4. (2, 12) + (2,): останні виміри 12 і 2 не збігаються і жоден не дорівнює 1 — помилка. Після reshape(2, 1) маємо (2, 12) + (2, 1): 12 і 1 сумісні, 2 і 2 рівні — broadcasting розтягує стовпець поправок на всі 12 місяців.
""")
