import pandas as pd

print("Практика 2. Варіант 1 — Київ")

months = ["Січ", "Лют", "Бер", "Кві", "Тра", "Чер", "Лип", "Сер", "Вер", "Жов", "Лис", "Гру"]
df = pd.DataFrame({
    "month": months,
    "avg_temp": [-4, -3, 2, 9, 15, 19, 21, 20, 14, 8, 2, -2],
    "precip": [38, 36, 38, 41, 53, 74, 84, 69, 47, 36, 44, 44],
})
print(df.head())
print(df.shape)
print(df.dtypes)

print("\nЗавдання 1")
print(df)
print(df.index)
df = df.set_index("month")
print(df.loc["Лип"])
df = df.reset_index()
print(df.index)

print("\nЗавдання 2")
df.index = range(1, 13)
print(df.loc[1:6, "avg_temp"])
print(df.iloc[0:6, 1])
print(".loc[1:6] бере підписи 1..6 включно, .iloc[0:6] — позиції 0..5 (кінець не включно). Тут підпис = позиція + 1, тому рядки збіглися; якби індекс був іншим (наприклад, 0..11 або перемішаний), .loc[1:6] повернув би інші рядки або іншу їх кількість.")
print(df[["month", "precip"]])

print("\nЗавдання 3")
print(df[df["avg_temp"] > 15])
print(df[(df["avg_temp"] > 10) & (df["precip"] > 50)])
print(df[(df["avg_temp"] < 0) | (df["precip"] > 80)])

print("\nЗавдання 4")
df["avg_temp_f"] = df["avg_temp"] * 9 / 5 + 32
df["is_wet"] = df["precip"] > df["precip"].mean()


def season(month_number):
    if month_number in (12, 1, 2):
        return "зима"
    elif month_number in (3, 4, 5):
        return "весна"
    elif month_number in (6, 7, 8):
        return "літо"
    else:
        return "осінь"


df["season"] = pd.Series(df.index, index=df.index).apply(season)
print(df)

print("""
Відповіді на питання:
1. .loc працює з підписами індексу і включає кінець зрізу, .iloc — з позиціями і кінець не включає. Якщо підписи не збігаються з позиціями (наприклад, індекс починається з 1 або має пропуски), .loc[1:6] і .iloc[0:6] повертають різні рядки, а за відсутніх підписів — і різну кількість.
2. and вимагає одного значення True/False для всього виразу, а Series із 12 значень не має однозначного логічного значення — виникає ValueError. Оператор & працює поелементно; дужки потрібні, бо & має вищий пріоритет, ніж > .
3. Формула (векторизована операція) застосовується до всього стовпця одразу і працює швидко; .apply() викликає власну функцію для кожного елемента окремо — це повільніше, але дозволяє довільну логіку з if/elif, як для сезону.
""")
