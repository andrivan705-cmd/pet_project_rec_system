import os
import pickle
from datetime import datetime
import hashlib
from typing import List, Tuple, Any

import pandas as pd
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import create_engine

# --- Pydantic Models ---
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

app = FastAPI()

# --- Data Loading ---

def batch_load_sql(query: str):
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        # Для тестов берём только первый чанк, но лучше загружать все
        break
    conn.close()
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_raw_features():
    logger.info("loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action='like'"""
    liked_posts = batch_load_sql(liked_posts_query)

    logger.info("loading posts features")
    posts_features = pd.read_sql(
        """SELECT * FROM public.posts_info_features_dl""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml"
    )

    logger.info("loading user features")
    user_features = pd.read_sql(
        """SELECT * FROM public.user_data""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml"
    )

    return [liked_posts, posts_features, user_features]


def load_model(model_version: str) -> Any:
    if os.environ.get("IS_LMS", "0") == "1":
        models_dir = os.environ.get("MODELS_DIR", ".")
    else:
        models_dir = "."

    model_filename = f"model_{model_version}.pkl"
    model_path = os.path.join(models_dir, model_filename)

    logger.info(f"Загрузка модели версии '{model_version}' из файла {model_path}...")

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f" Файл модели версии '{model_version}' не найден: {model_path}")
    except Exception as e:
        raise RuntimeError(f" Ошибка при загрузке модели версии '{model_version}': {e}") from e

    logger.success(f"Модель версии '{model_version}' успешно загружена")
    return model


# Глобальная загрузка данных и моделей при старте приложения
features = load_raw_features()
model_control = load_model("control")
model_test = load_model("test")

# --- A/B Splitting Logic ---

SALT = "my_salt"

def get_user_group(user_id: int) -> str:
    value_str = f"{user_id}{SALT}"
    hash_hex = hashlib.md5(value_str.encode()).hexdigest()
    percent = int(hash_hex[:8], 16) % 100
    return "control" if percent < 50 else "test"


# --- Feature Calculation & Recommendations ---

def calculate_features(
    user_id: int, dt: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Подготавливает данные для модели.
    Возвращает:
        user_features: признаки пользователя (без user_id)
        user_posts_features: объединенные признаки (посты + пользователь + время) для модели
        posts_text_topic: DataFrame с текстом и темой (для формирования ответа)
    """
    logger.info(f"Preparing features for user_id: {user_id}")

    # Извлекаем нужные DataFrame из глобального списка
    liked_posts_df = features[0]      # лайкнутые посты
    posts_df = features[1]            # признаки постов
    users_df = features[2]            # признаки пользователей

    # 1. Признаки пользователя
    user_features = users_df.loc[users_df.user_id == user_id]
    if user_features.empty:
        logger.warning(f"User {user_id} not found in user_features. Returning empty.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    user_features = user_features.drop("user_id", axis=1)

    # 2. Признаки постов (не удаляем text и topic)
    # В posts_df колонка 'index' может быть, удаляем её, чтобы не мешала
    posts_features = posts_df.drop(columns=["index"], errors="ignore")

    # Объединяем признаки пользователя с каждым постом
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_posts_features = posts_features.assign(**add_user_features)

    # 3. Сохраняем text и topic перед удалением
    posts_text_topic = user_posts_features[["text", "topic"]].copy()

    # 4. Удаляем text и topic из данных для модели
    user_posts_features = user_posts_features.drop(columns=["text", "topic"], errors="ignore")

    # 5. Добавляем временные признаки
    user_posts_features["hour"] = dt.hour
    user_posts_features["month"] = dt.month

    # 6. Устанавливаем post_id как индекс
    if "post_id" in user_posts_features.columns:
        user_posts_features = user_posts_features.set_index("post_id")
    else:
        # Если post_id уже индекс, ничего не делаем
        if user_posts_features.index.name != "post_id":
            # Пытаемся найти колонку с id
            id_col = next((col for col in user_posts_features.columns if col.lower() in ["post_id", "id"]), None)
            if id_col:
                user_posts_features = user_posts_features.set_index(id_col)

    return user_features, user_posts_features, posts_text_topic


def get_recommended_feed(user_id: int, dt: datetime, limit: int) -> Response:
    # 1. Определяем группу пользователя
    user_group = get_user_group(user_id)
    logger.info(f"User {user_id} assigned to group: {user_group}")

    # 2. Выбираем модель
    model = model_control if user_group == "control" else model_test

    # 3. Получаем фичи
    user_features, user_posts_features, posts_text_topic = calculate_features(
        user_id=user_id, dt=dt
    )

    if user_posts_features.empty:
        return Response(recommendations=[], exp_group=user_group)

    # 4. Предсказания
    logger.info("Running model prediction...")
    try:
        predicts = model.predict_proba(user_posts_features)[:, 1]
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        return Response(recommendations=[], exp_group=user_group)

    user_posts_features["predicts"] = predicts

    # 5. Убираем уже лайкнутые посты
    logger.info("Filtering out liked posts...")
    liked_posts_df = features[0]  # из глобального списка
    if not liked_posts_df.empty:
        liked_posts_ids = liked_posts_df[liked_posts_df.user_id == user_id].post_id.values
        filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts_ids)]
    else:
        filtered_ = user_posts_features

    if filtered_.empty:
        return Response(recommendations=[], exp_group=user_group)

    # 6. Топ-N по вероятности
    recommended_posts = filtered_.sort_values("predicts")[-limit:].index

    # 7. Берём текст и тему
    recommended_info = posts_text_topic.reindex(recommended_posts).dropna()

    recommendations = [
        PostGet(id=int(idx), text=str(row["text"]), topic=str(row["topic"]))
        for idx, row in recommended_info.iterrows()
    ]

    return Response(
        recommendations=recommendations,
        exp_group=user_group,
    )


# --- FastAPI Endpoint ---

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(user_id: int, dt: str, limit: int = 10) -> Response:
    """
    Эндпоинт для получения рекомендаций.
    Параметры:
        user_id: ID пользователя (int)
        dt: Время запроса в формате 'YYYY-MM-DD HH:MM:SS' (str)
        limit: Количество рекомендаций (int, default=10)
    """
    try:
        dt_obj = datetime.fromisoformat(dt.replace(' ', 'T'))
    except ValueError as e:
        logger.error(f"Invalid datetime format: {dt}. Error: {e}")
        raise ValueError(f"Invalid datetime format: {dt}")

    return get_recommended_feed(user_id=user_id, dt=dt_obj, limit=limit)