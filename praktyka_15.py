import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare

print("Практика 15. Варіант 1 — Київ, середньорічна 9.5°C, амплітуда 14°C")

alpha = 0.05
np.random.seed(42)
base_temp = 9.5
amplitude = 14
city = "Київ"


def season(month):
    if month in (12, 1, 2):
        return "зима"
    if month in (3, 4, 5):
        return "весна"
    if month in (6, 7, 8):
        return "літо"
    return "осінь"


rows = []
for year in [2021, 2022, 2023, 2024]:
    for month in range(1, 13):
        seasonal = amplitude * np.cos((month - 7) / 12 * 2 * np.pi)
        noise = np.random.normal(0, 1.0)
        temp = round(base_temp + seasonal + noise, 1)
        diff = temp - base_temp
        if diff < -3:
            norm_cat = "холодніше"
        elif diff > 3:
            norm_cat = "тепліше"
        else:
            norm_cat = "звичайно"
        rows.append({
            "місто": city, "рік": year, "місяць": month,
            "температура": temp, "сезон": season(month),
            "відхилення_від_норми": norm_cat,
        })

climate = pd.DataFrame(rows)
print(climate.shape)

print("\nЗавдання 1")
order_s = ["зима", "весна", "літо", "осінь"]
order_n = ["холодніше", "звичайно", "тепліше"]
table = pd.crosstab(climate["сезон"], climate["відхилення_від_норми"]).reindex(index=order_s, columns=order_n, fill_value=0)
print(table)
print("Найчастіші комбінації: 'зима' — 'холодніше' і 'літо' — 'тепліше' (по 12 разів, тобто всі спостереження цих сезонів). Це очікувано: категорія відхилення визначається сезонним коливанням навколо 9.5°C, тож улітку завжди тепліше, а взимку — холодніше.")
table_norm = pd.crosstab(climate["сезон"], climate["відхилення_від_норми"], normalize="index").reindex(index=order_s, columns=order_n, fill_value=0)
print(table_norm.round(3))
print("Тепер у кожній клітинці частка від усіх спостережень сезону, тому сума рядка дорівнює 1. Це дозволяє порівнювати структуру сезонів між собою незалежно від того, скільки спостережень у кожному.")

print("\nЗавдання 2")
chi2, pvalue, dof, expected = chi2_contingency(table)
expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
print(f"chi2 = {chi2:.3f}, p-value = {pvalue:.2e}, dof = {dof}")
print(expected_df.round(2))
if pvalue < alpha:
    print("p < 0.05 → відхиляємо H0 про незалежність: сезон і відхилення від норми пов'язані.")
else:
    print("p ≥ 0.05 → немає підстав відхилити H0 про незалежність.")
contrib = (table - expected_df) ** 2 / expected_df
cell = contrib.stack().idxmax()
print("Внесок клітинок у χ²:")
print(contrib.round(2))
print(f"Найбільша розбіжність — у клітинці '{cell[0]}' / '{cell[1]}': спостережено {table.loc[cell]}, очікувано {expected_df.loc[cell]:.2f}.")

print("\nЗавдання 3")
low = expected_df.stack()
low = low[low < 5]
print("Клітинки з очікуваною частотою < 5:")
print(low.round(2))
print(f"Таких клітинок {len(low)} з {expected_df.size} — увесь стовпець 'звичайно' (очікувано по 2.0), умова застосовності порушена.")
print("Об'єднуємо 'холодніше' і 'звичайно' в одну категорію 'не тепліше'.")
climate["тепло"] = np.where(climate["відхилення_від_норми"] == "тепліше", "тепліше", "не тепліше")
table2 = pd.crosstab(climate["сезон"], climate["тепло"]).reindex(order_s)
chi2_2, p_2, dof_2, exp_2 = chi2_contingency(table2)
print(table2)
print(pd.DataFrame(exp_2, index=table2.index, columns=table2.columns).round(2))
print(f"chi2 = {chi2_2:.3f}, p-value = {p_2:.2e}, dof = {dof_2}")
print("Усі очікувані частоти ≥ 5:", bool((exp_2 >= 5).all()))
print("Після об'єднання умова виконана, а висновок не змінився: p-value значно менше 0.05, зв'язок між сезоном і відхиленням від норми статистично значущий. Статистика χ² зменшилась, бо зникли клітинки з малими E, які давали великий внесок, — тепер висновок формально коректний.")

print("\nЗавдання 4")
observed = climate["сезон"].value_counts().reindex(order_s)
print(observed)
res = chisquare(f_obs=observed, f_exp=[len(climate) / 4] * 4)
print(f"chi2 = {res.statistic:.3f}, p-value = {res.pvalue:.4f}")
print("Кожен сезон має рівно 12 спостережень (3 місяці × 4 роки), тож статистика дорівнює 0, а p-value = 1. Це очікувано: розподіл сезонів задано структурою коду (повний перебір місяців), а не випадковістю, тому він точно рівномірний.")

print("\nЗавдання 5")
print("H0: 'холодніше' — 5/12 (≈41.7%), 'звичайно' — 2/12 (≈16.7%), 'тепліше' — 5/12 (≈41.7%).")
print("Обґрунтування: без шуму diff = 14·cos((місяць − 7)/12·2π). Межа |diff| ≤ 3 означає |cos| ≤ 0.214, що виконується лише для квітня і жовтня (cos = 0). Травень—вересень дають diff ≥ +7 (тепліше), листопад—березень — diff ≤ −7 (холодніше). Шум зі std = 1 не здатен перенести місяць через межу ±3, тому очікувана пропорція 5:2:5.")
obs_n = climate["відхилення_від_норми"].value_counts().reindex(order_n)
print(obs_n)
exp_n = np.array([5, 2, 5]) / 12 * len(climate)
res2 = chisquare(f_obs=obs_n, f_exp=exp_n)
print(f"chi2 = {res2.statistic:.3f}, p-value = {res2.pvalue:.4f}")
if res2.pvalue < alpha:
    print("p < 0.05 → відхиляємо H0: фактичні частки категорій суттєво відрізняються від заявлених.")
else:
    print("p ≥ 0.05 → немає підстав відхилити H0: фактичний розподіл узгоджується із заявленою пропорцією (але це не доводить, що вона точна). Тут частоти збіглися з очікуваними точно (20/8/20), бо межа ±3°C далеко від реальних відхилень місяців і шум їх не перекидає.")

print("""
Відповіді на питання:
1. Різниця (O − E) показує розбіжність у клітинці; квадрат робить усі розбіжності додатними, щоб вони не компенсували одна одну. Ділення на E нормує розбіжність на очікуваний масштаб: відхилення на 5 при E = 10 значно суттєвіше, ніж при E = 1000.
2. E = (сума рядка × сума стовпця) / загальна сума. За незалежності P(рядок і стовпець) = P(рядок) · P(стовпець); оцінюючи ці ймовірності частками в межових сумах і множачи на n, отримуємо саме цю формулу.
3. Велике p-value означає лише, що дані не суперечать H0 — розбіжність могла виникнути випадково. Це не доказ незалежності: зв'язок може існувати, але бути слабким або вибірка надто мала, щоб його виявити.
4. Статистика χ² має розподіл хі-квадрат лише наближено, і при малих E наближення погане — p-value стає неточним. Треба об'єднати категорії, збільшити вибірку або використати точний критерій (наприклад, Фішера для 2×2).
""")
