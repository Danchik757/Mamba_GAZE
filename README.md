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
- `scripts/build_ubunt_test1_bundle.sh`: локально собирает self-contained `test1` bundle для переноса на другую машину.
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

- `REPO_ROOT=/mnt/hd2/29d_kon/projects/Rendering/MAMBA_GAZE`
- `ENV_ROOT=/mnt/hd2/29d_kon/environments`
- `VIDEOS_DIR=/mnt/hd2/29d_kon/projects/Rendering/Mamba_1/non_textured_videos`
- `JSON_DIR=/mnt/hd2/29d_kon/projects/Rendering/Mamba_1/logs/non_mvp_data`
- `MESH_DIR=/mnt/hd2/29d_kon/projects/Rendering/Dataset/MeshMambaSaliency/MeshFile/non_texture`
- `GT_DIR=/mnt/hd2/29d_kon/projects/Rendering/Dataset/MeshMambaSaliency/SaliencyMap/non_texture`
- `GAZE_CSV_DIR=/mnt/hd2/29d_kon/projects/Rendering/MAMBA_GAZE/data/csv_for_models/MeshMamba_non_texture`
- `OUTPUT_DIR=/mnt/hd2/29d_kon/projects/Rendering/MAMBA_GAZE/run_outputs`

Важно:
- Репозиторий и входные `CSV` теперь предполагаются в одном каталоге на сервере.
- Внешними остаются только уже существующие каталоги с `MeshFile`, `SaliencyMap` и `logs/non_mvp_data`.

### Создание окружения на сервере

```bash
cd /mnt/hd2/29d_kon/projects/Rendering/MAMBA_GAZE
bash scripts/create_conda_env.sh
```

Это создаст окружение в:

```text
/mnt/hd2/29d_kon/environments/meshmamba_gaze
```

### Запуск Aquarium на сервере

```bash
cd /mnt/hd2/29d_kon/projects/Rendering/MAMBA_GAZE
bash scripts/run_aquarium_server.sh --device cuda:0 --precompute-all-frames --ray-batch-size 128
```

### Запуск произвольной модели

```bash
cd /mnt/hd2/29d_kon/projects/Rendering/MAMBA_GAZE
bash scripts/run_model_server.sh Aquarium_Deep_Sea_Diver_v1_L1 --device cuda:0
```

## Self-Contained `test1` для другой машины

Если нужно перенести только одну модель со всеми необходимыми файлами на машину без исходного репозитория и датасета:

```bash
cd /Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE
bash scripts/build_ubunt_test1_bundle.sh
```

По умолчанию это соберет bundle для `Aquarium_Deep_Sea_Diver_v1_L1`:
- рабочая директория: `/tmp/test1`
- архив для переноса: `/tmp/Aquarium_Deep_Sea_Diver_v1_L1_test1_bundle.tar.gz`

На новой машине:

```bash
mkdir -p /home/ubu/Documents/GAZE
tar -xzf /home/ubu/Documents/GAZE/Aquarium_Deep_Sea_Diver_v1_L1_test1_bundle.tar.gz -C /home/ubu/Documents/GAZE
cd /home/ubu/Documents/GAZE/test1
CONFIG_PATH=configs/test1_local.env bash scripts/create_conda_env.sh
CONFIG_PATH=configs/test1_local.env bash scripts/run_model_server.sh Aquarium_Deep_Sea_Diver_v1_L1 --device cpu --precompute-all-frames
```

Для sweep на новой машине:

```bash
cd /home/ubu/Documents/GAZE/test1
CONFIG_PATH=configs/test1_local.env bash scripts/sweep_aquarium_server.sh --device cpu
```
