# MeshMamba Gaze Projection

Пайплайн переносит `2D gaze`-траектории из `csv_for_models/MeshMamba_non_texture` на поверхность меша MeshMamba в `face-level` формате.

Что делает код:
- читает `CSV` с траекториями `t/x/y` по каждому участнику;
- читает `OBJ` меша и `JSON` с параметрами рендера;
- переводит экранные точки в лучи камеры;
- находит пересечение луча с треугольниками меша;
- накапливает `face hits`;
- сглаживает карту по графу смежности граней;
- нормализует карту каждого участника в `sum=1`;
- усредняет карты между участниками;
- сохраняет итоговую карту в формате `one value per line`, совместимом с GT MeshMamba;
- считает метрики против GT.

Ключевые договоренности для этой версии:
- используется `face-level`, потому что GT MeshMamba задан по граням;
- используются **все gaze-точки**, без отдельной детекции фиксаций;
- нормализация: сначала для каждого участника, затем агрегирование между участниками;
- основные метрики: `CC`, `SIM`, `KLD`, `MSE/SE`, дополнительно `Spearman`, `Cosine`;
- `NSS` и `AUC_Judd` считаются как proxy-метрики по top-percentile маскам GT, так как исходных fixation masks в релизе нет.

## Структура

- `run_meshmamba_gaze.py`: основной CLI-скрипт.
- `build_meshmamba_mapping.py`: строит JSON-маппинг имен между CSV / OBJ / JSON / GT.
- `mamba_gaze/`: модули загрузки данных, геометрии, метрик и самого пайплайна.
- `configs/lab_graphicon_server.env`: серверный конфиг под `lab.graphicon.ru`.
- `scripts/create_conda_env.sh`: создание `conda`-окружения на сервере.
- `scripts/run_model_server.sh`: запуск модели на сервере через конфиг.
- `scripts/run_aquarium_server.sh`: готовый запуск для `Aquarium_Deep_Sea_Diver_v1_L1`.
- `environment.server.yml`: минимальная `conda`-спека.

## Зависимости

Минимальные зависимости:

```bash
pip install -r requirements.txt
```

На сервере для ускорения нужен `torch` с `CUDA`.

## Git

Каталог подготовлен как отдельный репозиторий. После `git init` можно:

```bash
git add .
git commit -m "Initial MeshMamba gaze projection pipeline"
git remote add origin <REMOTE_URL>
git push -u origin codex/mamba-gaze-init
```

## Пример: одна модель

```bash
python run_meshmamba_gaze.py \
  --model Aquarium_Deep_Sea_Diver_v1_L1 \
  --device auto \
  --smoothing-steps 8 \
  --smoothing-alpha 0.6
```

Результаты будут в:

```text
./outputs/Aquarium_Deep_Sea_Diver_v1_L1/
```

Основные файлы:
- `aggregate_face_saliency_sum.csv`
- `aggregate_face_saliency_max.csv`
- `metrics_vs_gt.json`
- `participant_summary.csv`
- `run_summary.json`

## Легкий smoke-test

Без тяжелого прогона:

```bash
python run_meshmamba_gaze.py \
  --model Aquarium_Deep_Sea_Diver_v1_L1 \
  --device cpu \
  --max-participants 1 \
  --max-points-per-participant 8 \
  --ray-batch-size 8 \
  --no-precompute-all-frames
```

## Batch-режим

Скрипт уже подготовлен к переносу на сервер: пути можно передать через CLI. Для полного датасета стоит:
- сначала построить маппинг имен;
- затем запускать `run_meshmamba_gaze.py` по моделям;
- на GPU можно поднять `--ray-batch-size` и включить `--precompute-all-frames`.

## Замечание по именам

В MeshMamba есть расхождения между именами CSV и GT. Скрипт `build_meshmamba_mapping.py` строит явный JSON-маппинг и помечает:
- `exact`
- `canonical`
- `ambiguous`
- `missing`

Это нужно для последующего batch-запуска на всем наборе.

## Сервер `lab.graphicon.ru`

В репозитории уже зашит серверный конфиг `configs/lab_graphicon_server.env` с такими предположениями:

- `REPO_ROOT=/hd2/projects/Rendering/Mamba_1/MAMBA_GAZE`
- `ENV_ROOT=/hd2/environments`
- `VIDEOS_DIR=/hd2/projects/Rendering/Mamba_1/non_textured_videos`
- `JSON_DIR=/hd2/projects/Rendering/Mamba_1/logs/non_mvp_data`
- `MESH_DIR=/hd2/projects/Rendering/Dataset/MeshMambaSaliency/MeshFile/non_texture`
- `GT_DIR=/hd2/projects/Rendering/Dataset/MeshMambaSaliency/SaliencyMap/non_texture`
- `GAZE_CSV_DIR=/hd2/projects/Rendering/Mamba_1/csv_for_models/MeshMamba_non_texture`

Важно:
- `GAZE_CSV_DIR` я поставил как рабочее предположение. Если CSV лежат в другом месте, поменяй одну строку в конфиге.
- Если сервер видит диск не как `/hd2/...`, а как `/mnt/hd2/29d_kon/projects/...`, префикс тоже нужно поправить в конфиге.

### Создание окружения на сервере

```bash
cd /hd2/projects/Rendering/Mamba_1/MAMBA_GAZE
bash scripts/create_conda_env.sh
```

Это создаст окружение в:

```text
/hd2/environments/meshmamba_gaze
```

### Запуск Aquarium на сервере

```bash
cd /hd2/projects/Rendering/Mamba_1/MAMBA_GAZE
bash scripts/run_aquarium_server.sh --device cuda:0 --precompute-all-frames --ray-batch-size 128
```

### Запуск произвольной модели

```bash
cd /hd2/projects/Rendering/Mamba_1/MAMBA_GAZE
bash scripts/run_model_server.sh Aquarium_Deep_Sea_Diver_v1_L1 --device cuda:0
```
