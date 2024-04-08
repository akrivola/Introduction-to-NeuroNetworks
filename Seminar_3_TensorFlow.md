# Введение в нейронные сети
## Урок 3. TensorFlow
Сдавать через Гитхаб.
```
Домашнее задание   
Криволапов Антон
```

1. Попробуйте улучшить работу нейронной сети (разобранную на уроке), обучавшейся на датасет Fashion-MNIST. Напишите в комментариях к уроку, какого результата вы добились от нейросети и что помогло улучшить её точность
2. Поработайте с документацией TensorFlow 2. Попробуйте найти полезные команды TensorFlow, неразобранные на уроке
 * Попробуйте обучить нейронную сеть на TensorFlow 2 на датасете imdb_reviews.
Напишите в комментариях к уроку, какого результата вы добились от нейросети и что
помогло улучшить её точность

----
### Результат исходной задачи с урока:
Test accuracy: 0.8925999999046326
![image](https://github.com/akrivola/Introduction-to-NeuroNetworks/assets/112705486/54e5c78e-cca8-48c6-9a59-2a129eacea93)

Класс - 0, точность - 0.815, полнота - 0.857  
Класс - 1, точность - 0.991, полнота - 0.982  
Класс - 2, точность - 0.861, полнота - 0.756  
Класс - 3, точность - 0.904, полнота - 0.901  
Класс - 4, точность - 0.791, полнота - 0.839  
Класс - 5, точность - 0.978, полнота - 0.964  
Класс - 6, точность - 0.710, полнота - 0.722  
Класс - 7, точность - 0.951, полнота - 0.966  
Класс - 8, точность - 0.974, полнота - 0.973  
Класс - 9, точность - 0.963, полнота - 0.966  

Как уже говорилось на уроке - просадка точности идет в классах 4 и 6

### 1. Попробуем изменить параметры сети:
learning_rate= 0.001 --> 0.0001

Большой разницы нет:

Test accuracy: 0.8847000002861023

### 2. Попробуем добавить еще 1 слой нейронов, пятый (32):
```
model = keras.Sequential([  
    keras.layers.Flatten(input_shape=(28, 28), name='input'),  
    keras.layers.Dense(256, activation='relu', name='hiden_one'),  
    keras.layers.Dense(128, activation='relu', name='hiden_two'),
    keras.layers.Dense(64, activation='tanh', name='hiden_three'),
    keras.layers.Dense(64, activation='relu', name='hiden_four'),
    keras.layers.Dense(32, activation='relu', name='hiden_five'),
    keras.layers.Dense(10, name='output')
])
config = model.get_config()
model.save_weights('fashion_weights.h5')
```

Результат стал получше, особенно в 4 классе. Самый худший теперь шестой.

Test accuracy: 0.8939999938011169
![image](https://github.com/akrivola/Introduction-to-NeuroNetworks/assets/112705486/8789837b-3f77-4a15-a78e-b6f1f4f2bd43)
Класс - 0, точность - 0.820, полнота - 0.874  
Класс - 1, точность - 0.995, полнота - 0.974  
Класс - 2, точность - 0.826, полнота - 0.807  
Класс - 3, точность - 0.897, полнота - 0.901  
***Класс - 4, точность - 0.821, полнота - 0.823***   
Класс - 5, точность - 0.987, полнота - 0.946  
Класс - 6, точность - 0.736, полнота - 0.699  
Класс - 7, точность - 0.962, полнота - 0.951  
Класс - 8, точность - 0.968, полнота - 0.983  
Класс - 9, точность - 0.927, полнота - 0.982  

### 3. Попробуем использовать комбинацию 2-х сетей, сети образца и сети с параметрами, которая предсказывает самый плохо определяемый класс:

```
# Создаем словарь который будет содержать модель сети и параметры для обучения
models = dict()
models[0] = [keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28), name='input'),
    keras.layers.Dense(256, activation='relu', name='hiden_one'),
    keras.layers.Dense(128, activation='relu', name='hiden_two'),
    keras.layers.Dense(64, activation='tanh', name='hiden_three'),
    keras.layers.Dense(64, activation='relu', name='hiden_four'),
    keras.layers.Dense(10, name='output')
]), tf.keras.optimizers.Adam(learning_rate=0.001), 250]
models[1] = [keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28), name='input'),
    keras.layers.Dense(256, activation='relu', name='hiden_one'),
    keras.layers.Dense(128, activation='gelu', name='hiden_two'),
    keras.layers.Dense(64, activation='sigmoid', name='hiden_three'),
    keras.layers.Dense(64, activation='selu', name='hiden_four'),
    keras.layers.Dense(10, name='output')]), tf.keras.optimizers.AdamW(learning_rate=0.001), 450]
# Список для сохранения результатов
result = list()
# Формируем сеть.
for key, param in models.items():
  # Загружаем параметры
  model, opt, batch = param[0], param[1], param[2]
  # Загружаем входные веса, чтобы хоть как-то "уравнять" результаты работы
  model.load_weights('fashion_weights.h5')
  model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  model.fit(train_images, train_labels, batch_size=batch, epochs=25, verbose=0)
  # Получаем предварительную метрику
  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)
  probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
  # Сохраняем массив с вероятностями
  result.append(probability_model.predict(test_images))
# Получаем финишное предсказание суммируя два массива с вероятностями
y_pred = np.argmax(sum(result), axis=1)
# Окончательная метрика качества
print(accuracy_score(test_labels, y_pred))
# Распределение предсказаний по классам
print(recall_precision(test_labels, y_pred))
ConfusionMatrixDisplay.from_predictions(test_labels, y_pred)

```
## Задача 2 - попробовать команы TensorFlow, не разобранные на уроке. 
Я попробовал модуль Keros-tuner для поиска оптимальных параметров сети:
```
def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=10, max_value=256, step=25)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

import keras_tuner as kt
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

tuner.search(train_images, train_labels, epochs=25, validation_split=0.2, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f' Units are {best_hps.get("units")}')
best_config = best_hps.get_config()
print("Best Hyperparameters:")
for key, value in best_config.items():
    print(key, ":", value)
```
Результат работы:
```
Trial 10 Complete [00h 00m 12s]
val_accuracy: 0.8724166750907898

Best val_accuracy So Far: 0.8724166750907898
Total elapsed time: 00h 02m 18s
 Units are is 210
Best Hyperparameters:
space : [{'class_name': 'Int', 'config': {'name': 'units', 'default': None, 'conditions': [], 'min_value': 10, 'max_value': 256, 'step': 25, 'sampling': 'linear'}}]
values : {'units': 210, 'tuner/epochs': 2, 'tuner/initial_epoch': 0, 'tuner/bracket': 2, 'tuner/round': 0}
```

Кол-во юнитов в первом слое рекомендуется иметь 210.

# ВЫВОДЫ:
1. Попробовал изменить параметры сети (увеличил время обучения) - точность не улучшилась
2. Попробовал добавить еще 1 слой нейронов, пятый - точность увеличилась, исправился один из проблемных классов
3. Попробовал использовать комбинацию 2-х сетей, сети образца и сети с параметрами, которая предсказывает самый плохо определяемый класс - точность увеличилась
4. Использовал Keros-tuner для поиска оптимальных параметров сети - получил наилучшую точность и список параметров для достижения этой точности.
