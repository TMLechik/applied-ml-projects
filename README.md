# ASHRAE Energy Consumption Prediction

Прогнозирование почасового энергопотребления зданий на основе характеристик здания, погодных условий и временных признаков.

Проект выполнен на датасете соревнования **ASHRAE – Great Energy Predictor III** и охватывает полный ML-пайплайн: от анализа и оптимизации данных до feature engineering, классических моделей, MLP и ансамблирования.

## Задача

Необходимо предсказать значение `meter_reading` — почасовое потребление ресурса зданием.

В данных представлены четыре типа счётчиков:

- `0` — electricity;
- `1` — chilled water;
- `2` — steam;
- `3` — hot water.

Основные источники признаков:

- характеристики здания;
- назначение здания;
- тип счётчика;
- погодные наблюдения;
- время измерения.

В качестве основной метрики используется **RMSLE**.

Поскольку распределение `meter_reading` имеет выраженный правый хвост, модели обучаются на преобразованной целевой переменной:

```python
y = np.log1p(meter_reading)
```

RMSE в этом пространстве соответствует RMSLE на исходной целевой переменной.

---

## Данные

Используются данные соревнования:

**Kaggle:**  
https://www.kaggle.com/competitions/ashrae-energy-prediction

Основные таблицы:

| Dataset | Rows | Description |
|---|---:|---|
| `train.csv` | 20,216,100 | показания счётчиков |
| `test.csv` | 41,697,600 | test observations |
| `building_metadata.csv` | 1,449 | характеристики зданий |
| `weather_train.csv` | 139,773 | погодные наблюдения train |
| `weather_test.csv` | 277,243 | погодные наблюдения test |

После загрузки таблицы `train/test` объединяются с характеристиками зданий и погодными наблюдениями по `building_id`, `site_id` и `timestamp`.

---

## Стек

Основные библиотеки:

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- LightGBM
- XGBoost
- CatBoost
- PyTorch

---

## Memory Optimization

Исходные данные занимают более **3 GB RAM**, поэтому перед объединением таблиц была проведена оптимизация типов данных.

В частности:

- идентификаторы переведены в `uint8/uint16/uint32`;
- числовые погодные признаки — в `float32`;
- `meter_reading` — в `float32`;
- `primary_use` — в `category`;
- `timestamp` — в `datetime`.

Общий объём исходных таблиц удалось уменьшить примерно с:

```text
3047.6 MB
```

до:

```text
900.4 MB
```

Это существенно снижает требования к памяти при последующем merge и обучении моделей.

---

## EDA

В рамках исследовательского анализа были изучены:

- распределение `meter_reading`;
- распределение `log1p(meter_reading)`;
- потребление по типу счётчика;
- потребление по назначению здания;
- зависимость от площади здания;
- зависимость от количества этажей;
- суточные профили энергопотребления;
- погодные признаки;
- корреляции погоды с таргетом отдельно для разных типов счётчиков;
- нелинейные зависимости потребления от температуры, точки росы и скорости ветра;
- влияние осадков.

### Основные наблюдения

`meter_reading` имеет сильно скошенное вправо распределение с большим количеством экстремальных значений. Поэтому для обучения используется `log1p`-преобразование таргета.

Распределения энергопотребления существенно отличаются между типами счётчиков и назначениями зданий.

Площадь здания положительно связана с энергопотреблением, однако одного размера здания недостаточно для объяснения наблюдаемого разброса.

Также обнаружена выраженная суточная зависимость потребления, причём её форма различается для зданий разного назначения.

Линейные корреляции погодных признаков с таргетом относительно слабые, однако анализ сгруппированных значений показал наличие нелинейных зависимостей.

---

## Missing Values

Наиболее значительная доля пропусков наблюдается в:

- `floor_count`;
- `year_built`;
- `cloud_coverage`;
- `precip_depth_1_hr`;
- `wind_direction`;
- `sea_level_pressure`.

Стратегия обработки зависит от модели.

Для **Ridge, Decision Tree, XGBoost и MLP** числовые признаки заполняются медианой, рассчитанной только по обучающей выборке.

Категориальные признаки при необходимости заполняются наиболее частым значением.

**LightGBM** использует встроенную обработку пропущенных числовых значений.

Импутация выполняется после разбиения данных, что позволяет избежать использования статистик validation-части при обучении preprocessing.

---

## Validation Strategy

Данные имеют временную структуру, поэтому случайное перемешивание наблюдений может привести к утечке информации из будущего.

Для оценки моделей используется **time-aware validation** на основе `TimeSeriesSplit`.

Разбиение производится по уникальным значениям `timestamp`, поэтому все наблюдения одного момента времени полностью попадают либо в train, либо в validation.

На каждом следующем фолде обучающий временной диапазон расширяется, а validation всегда находится позже train.

Для вычислительно дорогих этапов — feature ablation и hyperparameter tuning — дополнительно используются уменьшенные выборки с сохранением временного порядка.

---

## Feature Engineering

Были исследованы несколько групп новых признаков.

### Temporal features

```text
month
is_weekend
```

### Cyclical features

```text
hour_sin
hour_cos
day_of_week_sin
day_of_week_cos
```

Циклическое кодирование позволяет корректно представить периодическую природу времени. Например, 23:00 и 00:00 должны находиться близко друг к другу в пространстве признаков.

### Building features

```text
log_square_feet
```

### Weather features

```text
temp_dew_diff
heating_degree
cooling_degree
```

где:

```text
temp_dew_diff = air_temperature - dew_temperature
heating_degree = max(18 - air_temperature, 0)
cooling_degree = max(air_temperature - 18, 0)
```

### Weather lag features

Дополнительно исследовались предыдущие значения температуры:

```text
air_temperature_lag_1
air_temperature_lag_24
air_temperature_lag_168
air_temperature_rolling_24
```

Rolling-признак строится только по прошлым наблюдениям:

```python
shift(1).rolling(24)
```

что исключает использование текущего или будущего значения температуры.

---

## Feature Ablation

Полезность engineered features проверялась отдельно на моделях различной природы:

- Ridge;
- Decision Tree;
- LightGBM;
- XGBoost.

Каждая группа признаков добавлялась к базовому набору отдельно, после чего сравнивалось изменение RMSLE.

Наиболее устойчивое улучшение показало циклическое представление часа.

После ablation и дополнительной совместной проверки для дальнейших экспериментов выбран единый набор:

```text
hour_sin
hour_cos
is_weekend
```
---

## Classical ML

В проекте были исследованы модели нескольких классов.

### Naive baseline

```text
DummyRegressor
```

### Linear models

```text
Ridge
SGDRegressor
```

### Tree-based model

```text
DecisionTreeRegressor
```

### Gradient Boosting

```text
LightGBM
XGBoost
CatBoost
```

Это позволяет сравнить линейные зависимости, одиночные деревья и ансамблевые алгоритмы на одной задаче.

---

## Hyperparameter Tuning

Для наиболее перспективных моделей выполнялся подбор гиперпараметров:

- Ridge;
- LightGBM;
- XGBoost.

Из-за размера исходного датасета search выполняется на уменьшенной временной выборке:

```text
2% train
10% validation period
```

Для перебора конфигураций используется `ParameterSampler`.

Подбираемые параметры включают:

### Ridge

```text
alpha
```

### LightGBM

```text
n_estimators
learning_rate
num_leaves
max_depth
min_child_samples
reg_lambda
```

### XGBoost

```text
n_estimators
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
reg_lambda
```

После поиска лучшие конфигурации используются в последующих экспериментах.

---

## Neural Network

Помимо классических алгоритмов реализована собственная MLP на PyTorch.

### Preprocessing

Числовые признаки:

```text
median imputation
→ StandardScaler
```

Категориальные признаки:

```text
building_id
meter
site_id
primary_use
```

кодируются с помощью обучаемых **embeddings**.

Такой подход позволяет избежать большого one-hot представления, особенно для `building_id`.

### Architecture

```text
Numerical features
        +
Categorical embeddings
        ↓
      Concatenate
        ↓
Linear(256)
BatchNorm
ReLU
Dropout(0.2)
        ↓
Linear(128)
ReLU
Dropout(0.1)
        ↓
Linear(64)
ReLU
        ↓
Linear(1)
```

Для обучения используется:

```text
Loss: MSE
Optimizer: AdamW
Learning rate: 1e-3
Weight decay: 1e-4
```

### Regularization

В сети используются сразу несколько методов регуляризации:

- Dropout;
- Batch Normalization;
- Weight Decay;
- Early Stopping.

Для анализа процесса обучения строятся графики:

- train / validation loss;
- train / validation RMSLE.

---

## Blending

Для проверки возможности улучшения качества были объединены:

```text
Ridge
LightGBM
XGBoost
MLP
```

Итоговое предсказание blending представляет собой взвешенную сумму предсказаний базовых моделей.

Веса подбираются на отдельном временном `meta_train` с ограничениями:

---

## Stacking

Также реализован **holdout stacking**.

Схема:

```text
base_train
    ↓
Ridge / LightGBM / XGBoost / MLP
    ↓
predictions on meta_train
    ↓
Ridge meta-model
```

Базовые модели не обучаются на `meta_train`, поэтому meta-model получает out-of-sample предсказания.

После выбора ансамбля базовые модели могут быть переобучены на всём train, а зафиксированная meta-model используется для объединения их test-предсказаний.

---

## Итоговые результаты
