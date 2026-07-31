# LCNN для обнаружения синтезированной речи

Проект реализует контрмеру на основе Light CNN для обнаружения синтезированной
и преобразованной речи в разделе Logical Access (LA) датасета ASVspoof 2019.

- Автор: **Маевский Матвей Максимович**
- Электронная почта: [mmmaevskiy@edu.hse.ru](mailto:mmmaevskiy@edu.hse.ru)

Проект выполнен на основе
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
в рамках домашнего задания по защите систем распознавания речи от спуфинг-атак.

## Результат

Лучший checkpoint показал следующий результат на полной evaluation-выборке
ASVspoof 2019 LA:

| Метрика | Результат |
| --- | ---: |
| Количество записей | 71 237 |
| EER | **4.63%** |

[Логи обучения в Weights & Biases](https://wandb.ai/matmath-work-hse-university/anti-spoofing/runs/3sm9phah?nw=nwusermatmathwork).

## Метод

### Входные признаки

Каждая аудиозапись преобразуется в логарифм спектра мощности с помощью
детерминированного STFT со следующими параметрами:

- размер FFT — 1724;
- длина окна Blackman — 1724;
- шаг окна — 130;
- `center=False`;
- используются первые 600 временных кадров;
- если кадров меньше 600, спектрограмма дополняется нулями справа.

Для одной записи получается тензор формы `[1, 863, 600]`. После объединения
объектов в батч вход модели имеет форму `[B, 1, 863, 600]`. Для обучения,
валидации и inference используется одно и то же детерминированное
преобразование.

### Модель

Контрмера основана на Light CNN с активациями Max-Feature-Map. Модель содержит
девять свёрточных слоёв, четыре операции max pooling, 80-мерное внутреннее
представление и линейный классификатор на два класса. Dropout расположен перед
последним слоем Batch Normalization.

Используется следующее соответствие меток:

```text
spoof     -> 0
bonafide  -> 1
```

В качестве результата для каждой записи сохраняется вероятность класса
`bonafide`. Чем выше значение, тем больше уверенность модели в том, что речь
является настоящей.

### Параметры обучения

| Параметр | Значение |
| --- | --- |
| Функция потерь | Cross-entropy |
| Оптимизатор | Adam |
| Начальный learning rate | `3e-4` |
| Параметры Adam | `(0.9, 0.999)` |
| Размер батча | 8 |
| Количество эпох | 15 |
| Scheduler | StepLR, уменьшение в `0.5` раза каждые 10 эпох |
| Выбор checkpoint | Минимальный EER на development-выборке |
| Seed | 10 |

Параметр `epoch_len` равен `null`, поэтому одна эпоха соответствует полному
проходу по обучающей выборке. Лучший checkpoint выбирается только по
`dev_EER`; evaluation EER не используется как monitor. Но итоговый best_model берется по evalutaion EER, так как преподаватель сказал, что так можно сделать в этой задаче в качестве исключения.

## Структура проекта

```text
.
├── train.py                   # запуск обучения
├── inference.py               # оценка модели и создание CSV
├── requirements.txt
└── src
    ├── configs                # Hydra-конфигурации экспериментов
    ├── datasets               # чтение протоколов и аудиофайлов ASVspoof
    ├── loss                   # cross-entropy
    ├── metrics                # EER по полной выборке
    ├── model                  # LCNN и Max-Feature-Map
    ├── trainer                # циклы обучения и inference
    └── transforms             # детерминированное FFT-преобразование
```

## Установка

Рекомендуется использовать Python 3.10 или 3.11.

```bash
git clone https://github.com/mmatmath/anti-spoofing.git
cd anti-spoofing

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

При работе в Kaggle необходимо подключить
[ASVspoof 2019 Dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)
к ноутбуку. Конфигурации используют следующую структуру каталогов:

```text
/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA
├── ASVspoof2019_LA_train/flac
├── ASVspoof2019_LA_dev/flac
├── ASVspoof2019_LA_eval/flac
└── ASVspoof2019_LA_cm_protocols
```

Если датасет расположен по другому пути, необходимо изменить `audio_dir` и
`protocol_path` в соответствующих файлах `src/configs/datasets/`.

## Запуск

### Проверка на одном батче

Для быстрой проверки всего pipeline предусмотрена отдельная конфигурация,
которая обрабатывает один train-батч за эпоху:

```bash
python train.py --config-name=lcnn_onebatch
```

### Полное обучение

```bash
python train.py
```

Checkpoint с минимальным development EER сохраняется по пути:

```text
saved/lcnn_cross_entropy_eer/model_best.pth
```

### Inference на evaluation-выборке

```bash
python inference.py \
  inferencer.from_pretrained=/kaggle/working/anti-spoofing/saved/lcnn_cross_entropy_eer/model_best.pth
```

Готовый файл для отправки сохраняется по пути:

```text
data/saved/asvspoof_eval/mmmaevskiy.csv
```

Файл не содержит заголовка. Каждая строка состоит из идентификатора записи и
оценки класса `bonafide`:

```csv
LA_E_2834763,0.483358234167099
LA_E_8877452,0.020565615966916084
```

После завершения inference также выводится EER, рассчитанный сразу по всей
evaluation-выборке, а не отдельно по батчам.

## Использованные материалы

- G. Lavrentyeva et al.,
  [«STC Antispoofing Systems for the ASVspoof2019 Challenge»](https://arxiv.org/abs/1904.05576),
  2019.
- X. Wang and J. Yamagishi,
  [«A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection»](https://arxiv.org/abs/2103.11326),
  2021.
- M. Todisco et al.,
  [«ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection»](https://arxiv.org/abs/1904.05441),
  2019.
- [Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template)
  использован как основа структуры проекта.

## Лицензия

В репозитории сохранена лицензия MIT исходного PyTorch Project Template.
Подробности приведены в файле [LICENSE](LICENSE).
