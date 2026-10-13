import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Практика 11. Варіант 1 — Київ, λ = 0.20, E[X] = 5.0 хв")

lam = 0.20
true_mean = 1 / lam
rng = np.random.default_rng(1)

print("\nЗавдання 1")
for n in [5, 50, 500, 20_000]:
    sample = rng.exponential(scale=1 / lam, size=n)
    print(f"n = {n:>6}: x̄ = {sample.mean():.4f}, |x̄ - E[X]| = {abs(sample.mean() - true_mean):.4f}")
print("Зі зростанням n різниця між x̄ і 5.0 загалом зменшується: при малих n відхилення може бути великим, при n = 20 000 воно мізерне. Зменшення не обов'язково монотонне на кожному кроці, але тенденція — наближення до E[X].")

print("\nЗавдання 2")
n_repeats = 10_000
means_n10 = rng.exponential(scale=1 / lam, size=(n_repeats, 10)).mean(axis=1)
means_n300 = rng.exponential(scale=1 / lam, size=(n_repeats, 300)).mean(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(means_n10, bins=40, density=True)
axes[0].set_title(f"n = 10, std = {means_n10.std():.3f}")
axes[1].hist(means_n300, bins=40, density=True)
axes[1].set_title(f"n = 300, std = {means_n300.std():.3f}")
for ax in axes:
    ax.set_xlabel("Вибіркове середнє x̄, хв")
    ax.set_ylabel("Густина")
fig.suptitle("Вибірковий розподіл середнього часу очікування, Київ")
plt.tight_layout()
plt.show()
print("При n = 10 гістограма ще помітно скошена вправо (довгий правий хвіст, як у вихідного експоненційного розподілу). При n = 300 вона симетрична й дзвоноподібна, і значно вужча.")

print("\nЗавдання 3")
rows = []
for n in [10, 30, 100, 300]:
    sample_means = rng.exponential(scale=1 / lam, size=(n_repeats, n)).mean(axis=1)
    rows.append({"n": n, "теоретична_SE": (1 / lam) / np.sqrt(n), "виміряна_std": sample_means.std()})
se_table = pd.DataFrame(rows)
se_table["відносна_різниця_%"] = (se_table["виміряна_std"] / se_table["теоретична_SE"] - 1) * 100
print(se_table.round(4))
print("Теоретична і виміряна SE збігаються з точністю до ~1–2% для всіх n — формула σ/√n підтверджується симуляцією.")

print("\nЗавдання 4")
se25 = (1 / lam) / np.sqrt(25)
se100 = (1 / lam) / np.sqrt(100)
print(f"SE(n=25) = {se25:.3f}, SE(n=100) = {se100:.3f}, відношення = {se25 / se100:.1f}")
print("Учетверо більша вибірка зменшила SE лише вдвічі, бо SE = σ/√n, а √4 = 2. Похибка спадає пропорційно кореню з n, тому кожне наступне зменшення похибки коштує дедалі більше спостережень.")

print("\nЗавдання 5")
means_small = rng.exponential(scale=1 / lam, size=(50, 10)).mean(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(means_n10, bins=40, density=True)
axes[0].set_title("n = 10, n_repeats = 10 000")
axes[1].hist(means_small, bins=15, density=True)
axes[1].set_title("n = 10, n_repeats = 50")
for ax in axes:
    ax.set_xlabel("Вибіркове середнє x̄, хв")
    ax.set_ylabel("Густина")
plt.tight_layout()
plt.show()
print("З n_repeats = 50 гістограма рвана й нерівна, форму розподілу важко впізнати. n визначає, яким є сам вибірковий розподіл x̄ (його ширину і форму), а n_repeats — скільки точок із цього розподілу ми бачимо. Чим більше повторів, тим точніше гістограма відтворює справжню форму, — так само як більша вибірка точніше описує будь-який розподіл.")

print("""
Відповіді на питання:
1. Закон великих чисел відповідає на питання 'куди прямує x̄': до E[X] зі зростанням n. ЦГТ відповідає на питання 'яка форма і розкид x̄': розподіл x̄ наближається до нормального з центром E[X] і стандартним відхиленням σ/√n.
2. Якби вихідні дані були нормальними, x̄ був би нормальним за будь-якого n, і ЦГТ нічого б не показала. Сильно скошений експоненційний розподіл дозволяє побачити саму суть теореми: нормальна форма x̄ виникає з усереднення, незалежно від форми вихідних даних.
3. Дисперсія середнього дорівнює σ²/n, а стандартне відхилення — корінь із дисперсії, тому SE = σ/√n. Щоб зменшити похибку вдвічі, треба вчетверо більше спостережень, а щоб у 10 разів — у 100 разів більше.
""")
