import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

data = np.array(
    [7.24, 6.24, 1.94, 0.07, 42.73, 9.10, 61.96, 17.49, 57.86, 14.31, 187.82, 1.88, 204.36, 11.38, 65.06, 124.61,
     151.84, 3.61, 16.66, 57.77, 113.99, 44.11, 20.49, 50.59, 22.30, 27.69, 27.88, 85.69, 97.43, 142.05, 127.90, 125.55,
     60.30, 121.01, 62.88, 31.43, 100.90, 79.12, 5.91, 17.33, 36.10, 9.17, 55.38, 84.49, 53.61, 5.02, 46.34, 34.28,
     8.79, 30.29, 21.06, 116.44, 88.14, 62.06, 3.31, 189.00, 45.41, 11.82, 120.05, 17.21, 83.74, 32.44, 97.85, 55.87,
     55.08, 361.48, 94.31, 91.32, 39.17, 66.85, 15.75, 139.17, 140.05, 49.50, 104.10, 182.87, 35.41, 138.48, 63.10,
     37.76, 33.01, 60.36, 5.43, 12.62, 42.23, 3.52, 88.04, 63.22, 3.71, 59.32, 60.65, 2.57, 2.96, 49.90, 12.61, 51.96,
     49.03, 50.31, 51.66, 89.56, 21.92, 13.83, 10.09, 5.48, 131.92, 30.50, 71.07, 111.32, 46.42, 134.68, 49.50, 79.65,
     3.37, 8.57, 87.83, 88.79, 23.21, 67.38, 133.83, 86.44, 19.07, 112.93, 57.38, 2.29, 22.36, 9.04, 10.75, 131.21,
     25.00, 12.94, 82.20, 43.57, 15.36, 24.99, 11.23, 193.46, 52.03, 72.30, 30.41, 248.87, 24.07, 6.23, 15.28, 3.86,
     121.36, 49.47, 23.05, 8.06, 30.82, 17.51, 42.42, 78.65, 19.18, 78.64, 4.97, 56.92, 10.53, 19.74, 74.35, 5.72,
     23.10, 24.54, 54.74, 95.76, 60.86, 38.43, 74.11, 66.96, 17.32, 73.50, 45.82, 55.19, 31.49, 18.88, 18.54, 47.59,
     124.64, 31.71, 27.87, 149.60, 164.85, 96.06, 70.96, 105.35, 11.48, 37.55, 50.70, 51.35, 124.56, 23.14, 13.40, 1.97,
     90.28, 82.43, 51.58, 121.68, 15.08, 10.99, 139.16, 86.92, 2.28, 102.18, 2.97, 17.00, 4.83, 81.74, 27.58, 14.23,
     32.91, 19.83, 40.98, 6.27, 25.13, 16.70, 24.92, 57.60, 56.00, 64.22, 21.86, 54.02, 19.27, 6.97, 12.27, 12.01,
     125.50, 22.15, 10.33, 4.60, 15.99, 17.65, 34.28, 111.73, 53.28, 16.15, 49.50, 15.79, 146.92, 6.66, 12.78, 189.66,
     26.12, 39.50, 10.17, 73.19, 3.84, 11.41, 15.84, 22.66, 70.91, 8.32, 61.69, 95.77, 41.46, 129.67, 3.17, 5.83, 50.52,
     17.13, 20.39, 13.31, 18.78, 16.78, 8.94, 90.10, 0.95, 4.25, 10.76, 56.99, 49.07, 186.87, 91.77, 88.65, 9.48, 4.21,
     56.42, 51.15, 6.38, 59.05, 56.49, 14.48, 54.26, 546.43, 60.54, 39.80, 47.57, 55.68, 18.46, 12.84, 146.48, 126.15,
     93.45, 66.07, 15.24, 42.69, 21.80, 2.82, 36.77, 140.76, 130.16, 213.524])
sample_counts = [300, 200, 100, 50, 20, 10]
np.random.seed(0)


def autocorrelation(X, k):
    n = len(X)
    mean_X = np.mean(X)
    numerator = np.sum((X[:n - k] - mean_X) * (X[k:] - mean_X))
    denominator = np.sqrt(np.sum((X - mean_X) ** 2) * np.sum((X[k:] - mean_X) ** 2))
    return numerator / denominator if denominator != 0 else 0


confidence_intervals = [0] * 6

def autocorrelation_analysis(sample,autocors,confidence_intervals):
    lags = range(n - 1)
    autocor_coefs = np.array([autocorrelation(sample, k) for k in lags])[1:]
    autocors.append(autocor_coefs)
    max_val = 1.96 / np.sqrt(n)
    confidence_intervals.append(max_val)
    print(f'Коэффициенты АК: {autocor_coefs[:10]}')
    # for i in range(10):
    #     ref_or_dev(f'autocor{i}', autocor_coefs[i])
    print(f'Граничное значние: {max_val:.3f}')
    if np.all(np.abs(autocor_coefs) < max_val):
        print('Последовательность является случайной.')
    else:
        print('Последовательность не является случайной.')
    return autocors,confidence_intervals


reference_values = {}


def ref_or_dev(key, value):
    if key not in reference_values:
        reference_values[key] = value
    else:
        relative_dev = abs((value - reference_values[key]) / reference_values[key])
        print(f" - отн. отклонение от эталона: {relative_dev * 100:.1f}%")


graphs = []
approximated_graphs = []


def approximate_distribution(cv, mean, sample, sample_size):
    print(f'Для N={sample_size} КВ={cv:.2f}.', end=' ')
    if cv < 0.01:
        print("Для аппроксимации используется нормальное распределение:")
        a, b = min(sample), max(sample)
        approximation = np.random.uniform(a, b, sample_size)
        return approximation

    elif np.isclose(cv, 1, atol=0.05):
        print("Для аппроксимации используется экспоненциальное распределение:")
        approximation = np.random.exponential(scale=mean, size=sample_size)
        return approximation

    elif cv < 1:
        print("Для аппроксимации используется распределение Эрланга:")
        k = int(round((1 / cv) ** 2))
        approximation = np.random.gamma(k, scale=mean / k, size=sample_size)
        return approximation

    elif cv > 1:
        print("Для аппроксимации используется гиперэкспоненциальное распределение:")
        lambda1 = 1 / mean
        lambda2 = lambda1 / 2
        approximation = np.concatenate([
            np.random.exponential(scale=1 / lambda1, size=int(sample_size / 2)),
            np.random.exponential(scale=1 / lambda2, size=int(sample_size / 2))
        ])
        return approximation


def mean_val(sample, n):
    mean = np.sum(sample) / n  # M(X)
    print(f'Мат ожидание: {mean:.2f}')
    return mean


def var_val(sample, mean):
    mean2 = np.sum(sample ** 2) / n  # M(X^2)
    var = mean2 - mean ** 2
    print(f'Дисперсия: {var:.2f}')
    return var


def SKO_val(var):
    SKO = np.sqrt(var)
    print(f'СКО: {SKO:.2f}')
    return SKO


def cv_val(mean, SKO):
    cv = SKO / mean
    print(f'Коэффициент вариации: {cv * 100:.1f}%')
    return cv


def confidence_interval(mean, n):
    for gamma in [0.9, 0.95, 0.99]:
        z = stats.norm.ppf((1 + gamma) / 2)  # Критическое значение Z
        error = z * (SKO / np.sqrt(n))
        lower = mean - error
        upper = mean + error
        print(f'Доверительный интервал уровня {gamma}: [{lower:.2f}, {upper:.2f}]')
        ref_or_dev(f'error{gamma}', lower * 2)

autocorrelations,confidence_intervals,autocorrelations_approximation,confidence_intervals_approximation=[],[],[],[]
for n in sample_counts:
    sample = data[:n]
    print(f'\nВыборка из {n} значений:')

    mean = mean_val(sample, n)
    ref_or_dev('mean', mean)

    var = var_val(sample, mean)
    ref_or_dev('var', var)

    SKO = SKO_val(var)
    ref_or_dev('SKO', SKO)

    cv = cv_val(mean, SKO)
    ref_or_dev('cv', cv)

    confidence_interval(mean, n)
    graphs.append(sample)

    is_increasing = all(x < y for x, y in zip(sample, sample[1:]))
    is_decreasing = all(x > y for x, y in zip(sample, sample[1:]))
    if is_increasing:
        print("Возрастающая последовательность")
    elif is_decreasing:
        print("Убывающая последовательность")
    else:
        print("Периодичная последовательность")

    autocorrelations,confidence_intervals = autocorrelation_analysis(sample,autocorrelations,confidence_intervals)
    approximation_sample = approximate_distribution(cv, mean, sample, n)
    approximated_graphs.append(approximation_sample)
    print("Числовые характеристики аппроксимации")
    approximation_mean = mean_val(approximation_sample, n)
    ref_or_dev('mean', approximation_mean)

    approximation_var = var_val(approximation_sample, approximation_mean)
    ref_or_dev('var', approximation_var)

    approximation_SKO = SKO_val(approximation_var)
    ref_or_dev('SKO', approximation_SKO)

    approximation_cv = cv_val(approximation_mean, approximation_SKO)
    ref_or_dev('cv', approximation_cv)

    confidence_interval(approximation_mean, n)

    correlation_coefficient = np.corrcoef(sample, approximation_sample)[0, 1]
    print(f"Коэффициент корреляции между исходной и сгенерированной последовательностями: {correlation_coefficient:.4f}")

    autocorrelations_approximation,confidence_intervals_approximation=autocorrelation_analysis(approximation_sample,autocorrelations_approximation,confidence_intervals_approximation)

fig, axs = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle("Графики числовой последовательности")
for i in range(len(graphs)):
    cur_plt = axs[i // 2][i % 2]
    cur_plt.plot(graphs[i], marker='o', linestyle='-', color='b')
    cur_plt.set_xlabel("Индекс")
    cur_plt.set_ylabel("Значение")
    cur_plt.grid()
plt.show()

fig, axs = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle("Гистограммы распределения частот")
for i in range(len(graphs)):
    cur_plt = axs[i // 2][i % 2]
    cur_plt.hist(graphs[i], bins=10, edgecolor='black', alpha=0.7)
    cur_plt.set_xlabel('Значения')
    cur_plt.set_ylabel('Частота')
plt.show()

fig, axs = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle('Аппроксимированная автокорреляционная функция')
for i in range(len(autocorrelations)):
    cur_plt = axs[i // 2][i % 2]
    cur_plt.stem(autocorrelations[i], markerfmt='o', linefmt='-', basefmt='k-')
    cur_plt.set_xlabel('Задержка (lag)')
    cur_plt.set_ylabel('Автокорреляция')
    cur_plt.axhline(y=confidence_intervals[i], color='r', linestyle='--')
    cur_plt.axhline(y=-confidence_intervals[i], color='r', linestyle='--')
plt.show()

fig, axs = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle("Графики аппроксимированной числовой последовательности")
for i in range(len(approximated_graphs)):
    cur_plt = axs[i // 2][i % 2]
    cur_plt.plot(approximated_graphs[i], marker='o', linestyle='-', color='b')
    cur_plt.set_xlabel("Индекс")
    cur_plt.set_ylabel("Значение")
    cur_plt.grid()
plt.show()

fig, axs = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle("Гистограммы распределения частот аппроксимации")
for i in range(len(approximated_graphs)):
    cur_plt = axs[i // 2][i % 2]
    cur_plt.hist(approximated_graphs[i], bins=10, edgecolor='black', alpha=0.7)
    cur_plt.set_xlabel('Значения')
    cur_plt.set_ylabel('Частота')
plt.show()

fig, axs = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle('Автокорреляционная функция апроксимации')
for i in range(len(autocorrelations_approximation)):
    cur_plt = axs[i // 2][i % 2]
    cur_plt.stem(autocorrelations_approximation[i], markerfmt='o', linefmt='-', basefmt='k-')
    cur_plt.set_xlabel('Задержка (lag)')
    cur_plt.set_ylabel('Автокорреляция')
    cur_plt.axhline(y=confidence_intervals_approximation[i], color='r', linestyle='--')
    cur_plt.axhline(y=-confidence_intervals_approximation[i], color='r', linestyle='--')
plt.show()

fig, axs = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle("Сравнение плотности распределения аппроксимации с гистограммой")
for i in range(len(graphs)):
    cur_plt = axs[i // 2][i % 2]
    cur_plt.hist(graphs[i], bins=15, density=True, edgecolor='black', alpha=0.6, label='Исходные данные')
    xmin, xmax = min(graphs[i].min(), approximated_graphs[i].min()), max(graphs[i].max(), approximated_graphs[i].max())
    x = np.linspace(xmin, xmax, len(approximated_graphs[i]))
    kde_approximation = stats.gaussian_kde(approximated_graphs[i])
    cur_plt.plot(x, kde_approximation(x), color='red', label='Аппроксимирующий закон', lw=2)
    cur_plt.set_xlabel('Значения')
    cur_plt.set_ylabel('Плотность')
    cur_plt.legend()
plt.show()