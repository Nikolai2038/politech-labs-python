# politech-labs-python

## Применение методов искусственного интеллекта для цифровой обработки сигналов

Используемая версия Python: 12.7. Подготовка Python:

```bash
python -m venv ./venv && \
./venv/bin/python -m pip install --upgrade pip && \
./venv/bin/pip install tensorflow Image
```

### Лабораторная работа 10

1. Создать и обучить модель:

    ```bash
    cd ./subject_01_ai-for-signals/lab_10_Python_150_digits && \
    { \
        ../../venv/bin/python ./classificatorR1.py; \
        cd -; \
    }
    ```

2. Протестировать модель:

    ```bash
    cd ./subject_01_ai-for-signals/lab_10_Python_150_digits && \
    { \
        ../../venv/bin/python ./test_1_CNN.py; \
        cd -; \
    }
    ```

### Лабораторная работа 11 (150 NNF)

1. Создать и обучить модель:

    ```bash
    cd ./subject_01_ai-for-signals/lab_11_Python_150_NNF_demo && \
    { \
        ../../venv/bin/python ./classificatorR1.py; \
        cd -; \
    }
    ```

2. Протестировать модель:

    ```bash
    cd ./subject_01_ai-for-signals/lab_11_Python_150_NNF_demo && \
    { \
        ../../venv/bin/python ./test_1_TNN.py; \
        cd -; \
    }
    ```

### Лабораторная работа 11 (1500 NNF)

1. Создать и обучить модель:

    ```bash
    cd ./subject_01_ai-for-signals/lab_11_Python_1500_NNF_demo && \
    { \
        ../../venv/bin/python ./classificatorR1.py; \
        cd -; \
    }
    ```

2. Протестировать модель:

    ```bash
    cd ./subject_01_ai-for-signals/lab_11_Python_1500_NNF_demo && \
    { \
        ../../venv/bin/python ./test_1_TNN.py; \
        cd -; \
    }
    ```
