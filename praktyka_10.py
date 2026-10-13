import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

print("Практика 10. Варіант 1 — Київ, липнева температура 21°C")

july_temp = 21

print("\nЗавдання 1")
np.random.seed(1)
daily_temps = np.random.normal(loc=july_temp, scale=2.5, size=30)
print(np.round(daily_temps, 2))
print(f"mean = {daily_temps.mean():.3f}, std = {daily_temps.std():.3f}")
print(f"Середнє вибірки {daily_temps.mean():.2f} близьке до 21, але не дорівнює йому — це випадкова вибірка з 30 значень.")

print("\nЗавдання 2")
mean, std = daily_temps.mean(), daily_temps.std()
dist = stats.norm(loc=mean, scale=std)
print(f"Підібраний розподіл: N({mean:.3f}, {std:.3f})")

print("\nЗавдання 3")
x = np.linspace(daily_temps.min() - 3, daily_temps.max() + 3, 200)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(daily_temps, bins=7, density=True, edgecolor="white", alpha=0.7, label="вибірка")
ax.plot(x, dist.pdf(x), color="crimson", linewidth=2, label="pdf N(mean, std)")
ax.set_xlabel("Денна температура, °C")
ax.set_ylabel("Густина")
ax.set_title("Денні температури липня, Київ: гістограма і нормальна крива")
ax.legend()
plt.show()
print("Крива загалом повторює форму гістограми: максимум біля середнього і спадання до країв. Окремі стовпці вищі чи нижчі за криву — для 30 значень такі відхилення природні.")

print("\nЗавдання 4")
p_1sigma = dist.cdf(mean + std) - dist.cdf(mean - std)
empirical = np.mean((daily_temps >= mean - std) & (daily_temps <= mean + std))
print(f"P(mean - std ≤ X ≤ mean + std) за моделлю = {p_1sigma:.4f}")
print(f"Фактична частка значень вибірки в цьому діапазоні = {empirical:.4f}")
print("За моделлю ймовірність точно 68.27% — це властивість будь-якого нормального розподілу, тому вона збігається з правилом 68-95-99.7. Фактична частка у вибірці може відрізнятися на кілька відсотків, бо 30 значень — лише одна випадкова реалізація, а не точний розподіл.")

print("\nЗавдання 5")
p95 = dist.ppf(0.95)
print(f"95-й процентиль = {p95:.2f} °C")
print(f"За підібраною моделлю 95% липневих днів мають температуру нижче {p95:.2f} °C, і лише 5% найспекотніших днів — вище. .cdf(x) за значенням дає накопичену ймовірність, а .ppf(p) навпаки — за ймовірністю дає значення: dist.cdf({p95:.2f}) = {dist.cdf(p95):.2f}.")

print("""
Відповіді на питання:
1. .pdf(x) повертає густину ймовірності в точці x — 'наскільки щільно' розташовані значення біля x. .cdf(x) повертає ймовірність P(X ≤ x) — яка частка значень не перевищує x.
2. Густина — не ймовірність, а ймовірність на одиницю довжини. Ймовірність — це площа під кривою, і вона дорівнює 1 загалом. Якщо розподіл дуже вузький (мале σ), крива має бути високою, щоб площа лишалася 1, тому pdf може перевищувати 1.
3. .cdf() приймає значення x і повертає ймовірність p = P(X ≤ x); .ppf() приймає ймовірність p і повертає значення x, для якого P(X ≤ x) = p. Тому ppf(cdf(x)) = x, і навпаки.
""")
