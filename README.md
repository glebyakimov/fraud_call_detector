Детектор телефонного мошенничества.

- **Вход**: `.wav`
- **Выход**: `label`
  - `0` — мошенник
  - `1` — не мошенник

Используется ASR (GigaAM, (от сбера конечно же)) для получения текста, затем гибридный классификатор (TF‑IDF + триггеры + MLP).

### Установка (Windows)

Нужны Python **3.12+** и FFmpeg.

Установка FFmpeg:

```powershell
winget install --id Gyan.FFmpeg -e
```

Установка зависимостей:

```powershell
python -m pip install -r requirements-asr.txt
```

### Запуск на одном файле

```powershell
python predict_one.py "path\to\file.wav"
```

Печатает только `0` или `1`.

### Запуск по папке

```powershell
python predict_folder.py "path\to\wav_folder" --out results.csv
```

CSV будет в формате:
`Название файла;label`

### Чекпоинт модели

По умолчанию используется чекпоинт:
`checkpoints/hybrid_from_train_plus_test`

Если нужно — можно указать другой, если вы хотите проверить на своих чекпоинтах.

```powershell
python predict_one.py file.wav --checkpoint checkpoints\...
python predict_folder.py wavs --out results.csv --checkpoint checkpoints\...
```

