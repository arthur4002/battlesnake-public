# Battlesnake ML Bot (PPO self-play)

Ветка `ML`: змейка на нейросети вместо эвристики. Обучение — PPO self-play
на GPU (A100, бюджет ~60–90 минут), инференс — лёгкий ONNX на CPU (Render
free tier укладывается в лимит 500 мс с запасом).

## Как это устроено

- `ml/rules.py` — точный движок стандартных правил Battlesnake
  (одновременные ходы, head-to-head, рост через дублирование хвоста,
  спавн еды 15%/ход, min food = 1).
- `ml/env.py` — векторизованное окружение self-play: N игр × 4 змейки,
  все места контролирует одна и та же сеть, обучение идёт на переходах всех
  четырёх (симметрия + 4× сэмпл-эффективность). Окружение шардируется по
  процессам (`--workers`), GPU делает батчевые форварды.
- `ml/encoding.py` — 12-канальное наблюдение 11×11 (голова/тело/хвост, головы
  врагов длиннее/короче, TTL занятых клеток, еда, health, разница длин) +
  маска заведомо смертельных ходов. **Один и тот же код** используется при
  обучении и при инференсе из JSON — рассинхрон исключён.
- `ml/model.py` — резидуальная CNN (8 блоков × 128 каналов, ~2.4M параметров,
  policy + value головы). На CPU — единицы миллисекунд на ход.
- `ml/train.py` — PPO (GAE, clip 0.2, bf16 на A100, маскирование действий,
  линейный декей lr и энтропии), ограничение по wall-clock, чекпойнты каждые
  5 минут, автоэкспорт в конце.
- `ml/inference.py` + `logic.py` — сервер: ONNX Runtime (или TorchScript),
  маска безопасности поверх политики, fallback на безопасный ход при любой
  ошибке. `backend.py` не меняется.

Награда: +1 победа, −1 смерть, +0.03 еда, +0.002 за выживание на ходу.

## 1. Обучение на A100

```bash
git clone -b ML https://github.com/Alex2034/battlesnake-public.git
cd battlesnake-public

python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # под ваш CUDA-драйвер
pip install -r requirements-train.txt

# 75 минут обучения (укладывается в бюджет 1–1.5 ч):
python -m ml.train --minutes 75 --num-envs 256 --workers 16
```

Полезные ручки:
- `--workers` — ставьте по числу физических ядер (env — CPU-bound);
- `--num-envs 384` — если ядер ≥ 32;
- `--resume ml/weights/ckpt_latest.pt` — продолжить обучение;
- `--minutes 60` — если время поджимает.

В логе смотрите `r/step` (средняя награда за шаг) — должна расти, и
`ent` (энтропия) — должна плавно падать. Чекпойнт `ml/weights/ckpt_latest.pt`
сохраняется каждые 5 минут, так что прервать обучение безопасно в любой момент.

По окончании в `ml/weights/` появятся:
- `ckpt_latest.pt` — чекпойнт (для resume/eval);
- `policy.onnx` — то, что использует сервер;
- `policy_ts.pt` — TorchScript-запаска.

Если обучение прервали и экспорт не отработал:
```bash
python -m ml.export --ckpt ml/weights/ckpt_latest.pt
```

Быстрая проверка силы (win rate против safe-random, база 25%):
```bash
python -m ml.eval --games 200
```
Хорошая модель после часа даёт >90%.

## 2. Локальный тест сервера

```bash
pip install -r requirements.txt
python backend.py
# в соседнем терминале, при установленном battlesnake CLI:
battlesnake play -W 11 -H 11 -n ml -u http://localhost:8000 -g solo -v
```

## 3. Деплой и скоринг на play.battlesnake.com

1. Закоммитьте веса в ветку (`policy.onnx` ~10 МБ — git ок):
   ```bash
   git add ml/weights/policy.onnx ml/weights/policy_ts.pt
   git commit -m "trained PPO policy" && git push origin ML
   ```
2. В [Render dashboard](https://dashboard.render.com): **New → Blueprint** →
   подключить репозиторий, выбрать ветку **ML** (или в существующем сервисе
   переключить branch на ML в Settings). Build:
   `pip install -r requirements.txt`, start:
   `gunicorn backend:app --bind 0.0.0.0:$PORT`.
3. Дождитесь деплоя, откройте URL — должен вернуться JSON с
   `"version": "ml-ppo-1.0"`.
4. На [play.battlesnake.com](https://play.battlesnake.com):
   - **Create Battlesnake** (или Edit существующей) → вставить Render URL;
   - создайте игру в **Play → Create Game**, добавьте свою змейку — проверка;
   - для рейтинга зайдите в **Compete → Leaderboards**, выберите арену
     (например Standard) и **Enroll** свою змейку — платформа сама будет
     запускать матчи и обновлять рейтинг. Прогресс виден на странице змейки.

Замечание: free tier Render засыпает после простоя; первый запрос после сна
может таймаутиться, и матч будет проигран. Для стабильного скоринга либо
платный инстанс, либо пингуйте URL раз в несколько минут (например, cron /
UptimeRobot).
