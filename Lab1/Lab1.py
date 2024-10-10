import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

with open('data') as f:
    data = np.array(list(map(float, f.readline().split(', '))))
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