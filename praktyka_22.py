import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.tsa.seasonal import seasonal_decompose

print("Практика 22. Варіант 1 — Київ, 9.5°C, амплітуда 14°C, тренд 0.04°C/рік, шум σ = 1.0")

alpha = 0.05
np.random.seed(1)
base_temp = 9.5
amplitude = 14
trend_per_year = 0.04
noise_scale = 1.0
city = "Київ"

n_years = 10
dates = pd.date_range(start="2015-01-01", periods=n_years * 12, freq="MS")
month = dates.month
years_elapsed = (dates.year - dates.year[0]) + (dates.month - 1) / 12

seasonal = amplitude * np.cos((month - 7) / 12 * 2 * np.pi)
trend = trend_per_year * years_elapsed
noise = np.random.normal(0, noise_scale, len(dates))

temp = np.round(base_temp + trend + seasonal + noise, 1)

climate_ts = pd.Series(temp, index=dates, name="температура")
print(climate_ts.head(12))
print(len(climate_ts))

print("\nЗавдання 1")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(climate_ts.index, climate_ts.values)
ax.set_xlabel("Місяць")
ax.set_ylabel("Температура, °C")
ax.set_title("Середньомісячна температура, Київ, 2015–2024")
plt.show()
print("На графіку чітко видно річний сезонний цикл: літні піки біля +23…+25°C і зимові мінімуми біля −4…−6°C, розмах ≈28°C. Повільного підйому рівня на око майже не видно: закладений тренд — лише 0.4°C за 10 років, що губиться на тлі сезонного розмаху і шуму σ = 1°C.")

print("\nЗавдання 2")
ma3 = climate_ts.rolling(window=3, center=True).mean()
ma12 = climate_ts.rolling(window=12, center=True).mean()
ma12_trailing = climate_ts.rolling(window=12).mean()
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
axes[0].plot(climate_ts.index, climate_ts.values, alpha=0.3, label="оригінал")
axes[0].plot(ma3.index, ma3.values, label="вікно = 3, центроване")
axes[0].plot(ma12.index, ma12.values, label="вікно = 12, центроване")
axes[0].set_ylabel("Температура, °C")
axes[0].set_title("Ковзне середнє з різними вікнами, Київ")
axes[0].legend()
axes[1].plot(ma12.index, ma12.values, label="вікно = 12, центроване")
axes[1].plot(ma12_trailing.index, ma12_trailing.values, label="вікно = 12, трейлінгове")
axes[1].set_xlabel("Місяць")
axes[1].set_ylabel("Температура, °C")
axes[1].set_title("Центроване проти трейлінгового ковзного середнього")
axes[1].legend()
plt.tight_layout()
plt.show()
shift = (ma12_trailing.shift(-6) - ma12).abs().max()
print(f"Максимальна різниця між трейлінговим, зсунутим на 6 місяців назад, і центрованим: {shift:.3f}°C")
print("Вікно 3 прибирає лише дрібний шум, сезонна хвиля лишається майже повністю. Вікно 12 дорівнює періоду сезонності: у кожне вікно потрапляє рівно один повний рік, і сезонні відхилення (тепле літо і холодна зима) взаємно гасяться — лишається тренд плюс згладжений шум. Вікно, не кратне 12, захоплювало б сезон нерівномірно.")
print("Трейлінгове вікно усереднює 12 попередніх місяців, тому його значення 'запізнюється' приблизно на пів вікна (≈6 місяців) і починається лише з 12-го місяця. Центроване значення відповідає середині вікна і не має цього лагу, але втрачає по 6 точок з обох кінців ряду. На графіку трейлінгова лінія — це центрована, зсунута вправо на ≈6 місяців.")

print("\nЗавдання 3")
decomposition = seasonal_decompose(climate_ts, model="additive", period=12)
fig = decomposition.plot()
fig.set_size_inches(10, 7)
plt.show()
tr = decomposition.trend.dropna()
print(f"Тренд: від {tr.iloc[0]:.2f} до {tr.iloc[-1]:.2f}°C, мін {tr.min():.2f}, макс {tr.max():.2f}")
seas = decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean()
print("Сезонна складова за місяцями:")
print(seas.round(2))
print(f"Найтепліший місяць: {seas.idxmax()} ({seas.max():+.2f}°C), найхолодніший: {seas.idxmin()} ({seas.min():+.2f}°C)")
resid_std = decomposition.resid.dropna().std()
print(f"std залишку: {resid_std:.3f}°C (закладено σ = {noise_scale})")
print("Тренд коливається в межах приблизно 9.2–10.3°C зі слабким зростанням, але не є рівною прямою: на рівні 0.04°C/рік він порівнянний з випадковими коливаннями річних середніх, тому виглядає хвилястим. Сезонна складова має пік у липні й мінімум у січні з розмахом близько ±14°C — точно як у формулі генерації (amplitude·cos з піком у місяці 7). Стандартне відхилення залишку трохи менше за закладений σ = 1, бо частину шуму 'вбирають' згладжений тренд і усереднений сезонний профіль.")

print("\nЗавдання 4")
t = np.arange(len(climate_ts))
known = decomposition.trend.notna().values
fit = linregress(t[known], decomposition.trend[known])
print(f"нахил тренду = {fit.slope:.4f} °C/місяць = {fit.slope * 12:.3f} °C/рік")
n_ahead = 12
t_future = np.arange(len(climate_ts), len(climate_ts) + n_ahead)
trend_forecast = fit.intercept + fit.slope * t_future
future_dates = pd.date_range(climate_ts.index[-1] + pd.offsets.MonthBegin(1), periods=n_ahead, freq="MS")
seasonal_forecast = seas.reindex(future_dates.month).values
forecast = pd.Series(trend_forecast + seasonal_forecast, index=future_dates, name="прогноз")
print(forecast.round(1))
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(climate_ts.index, climate_ts.values, label="історичні дані")
ax.plot(forecast.index, forecast.values, "o--", color="crimson", label="прогноз на 2025 рік")
ax.set_xlabel("Місяць")
ax.set_ylabel("Температура, °C")
ax.set_title("Прогноз середньомісячної температури, Київ")
ax.legend()
plt.show()
print("Застереження: прогноз — це лише продовження прямої і середнього сезонного профілю, підібраних на 2015–2024 роках. Він припускає, що нахил тренду і форма сезону не зміняться, а погода кожного місяця буде 'типовою'. Насправді конкретний місяць може відхилитися на кілька градусів (аномально тепла зима, хвиля спеки), а сам тренд залежить від глобальних кліматичних процесів, вулканічних викидів, явищ на кшталт Ель-Ніньо — нічого з цього модель не знає. Крім того, нахил оцінено на короткому ряді й він сам має велику невизначеність (див. Завдання 5). Тому числа прогнозу — орієнтир на рівні 'типового року', а не гарантія, і з кожним місяцем горизонту довіра до них менша.")

print("\nЗавдання 5")
deseason = climate_ts - decomposition.seasonal
t_years = np.arange(len(climate_ts)) / 12
trend_test = linregress(t_years, deseason.values)
print(f"нахил = {trend_test.slope:.4f} °C/рік, stderr = {trend_test.stderr:.4f}, p-value = {trend_test.pvalue:.4f}")
print("H0: істинний нахил тренду дорівнює нулю — реального потепління немає, видима зміна — шум. H1: нахил ≠ 0.")
ci_lo = trend_test.slope - 1.96 * trend_test.stderr
ci_hi = trend_test.slope + 1.96 * trend_test.stderr
print(f"95% ДІ для нахилу: ({ci_lo:.4f}, {ci_hi:.4f}) °C/рік")
if trend_test.pvalue < alpha:
    print("p < 0.05 → відхиляємо H0: тренд статистично підтверджений.")
else:
    print("p ≥ 0.05 → немає підстав відхилити H0: на 10 роках тренд статистично не підтверджений.")
if ci_lo <= 0 <= ci_hi:
    print("Інтервал містить 0 — дані сумісні як з відсутністю тренду, так і з потеплінням до верхньої межі інтервалу. Він також містить закладене значення 0.04°C/рік, тобто метод не суперечить справжньому тренду — просто його недостатньо, щоб відрізнити такий слабкий тренд від нуля.")
else:
    print("Інтервал не містить 0 — нахил статистично відмінний від нуля; він показує діапазон правдоподібних значень швидкості потепління.")

for years in [2, 3]:
    k = years * 12
    short = linregress(t_years[-k:], deseason.values[-k:])
    print(f"Останні {years} роки: нахил = {short.slope:+.3f} °C/рік, stderr = {short.stderr:.3f}, p = {short.pvalue:.3f}")
print("На короткому вікні оцінка нахилу помітно відрізняється від 10-річної і від закладених 0.04°C/рік (може бути навіть у кілька разів більшою або від'ємною), а її стандартна похибка в рази більша. Кілька випадково теплих чи холодних місяців наприкінці ряду легко створюють 'видимий тренд', але статистично підтвердити його на 24–36 точках ще важче.")
print(f"Висновок: для Києва (варіант 1) тренд потепління {'статистично підтверджений' if trend_test.pvalue < alpha else 'на 10 роках статистично не підтверджений'} — це спирається на p-value нахилу ({trend_test.pvalue:.3f}) і довірчий інтервал, а не на знак slope. Закладений тренд 0.04°C/рік (0.4°C за 10 років) занадто слабкий порівняно з шумом σ = 1°C, щоб надійно виявити його на такій довжині ряду.")

print("""
Відповіді на питання:
1. Методи Лекцій 12–16 припускають незалежні спостереження з одного розподілу. У часовому ряді сусідні значення залежні (автокореляція), а середнє змінюється з часом (тренд, сезонність), тому ряд не є вибіркою з одного розподілу — стандартні похибки й p-value 'звичайних' тестів стають хибними.
2. Замале вікно майже не згладжує — лишаються шум і сезонні коливання. Завелике вікно згладжує і корисну структуру (повороти тренду), а також втрачає більше точок на краях і дає більший лаг. Вікно варто брати рівним або кратним періоду сезонності (12 для місячних даних), щоб сезонна хвиля повністю гасилася.
3. Трейлінгове вікно усереднює поточний і попередні місяці, тому запізнюється приблизно на пів вікна. Центроване усереднює і минулі, і майбутні значення навколо точки — лагу немає, але для останніх місяців потрібні дані з майбутнього, яких у реальному часі ще немає.
4. Знак slope буде додатним чи від'ємним завжди, навіть для чистого шуму. Лише p-value (і довірчий інтервал) показують, чи такий нахил міг випадково виникнути за відсутності тренду; без цього 'тренд' може бути просто шумом, що на графіку виглядає як підйом.
""")
