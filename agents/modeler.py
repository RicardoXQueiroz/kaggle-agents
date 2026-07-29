"""Agente responsável por preparar features, treinar e validar modelos.

Baseline assume problema tabular (o caso mais comum em competições de
entrada). Para dados de imagem/texto, troque o estimador — a interface
com o orquestrador (recebe/devolve PipelineState) continua igual.
"""
import uuid
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, balanced_accuracy_score

from core.state import PipelineState


class _EnsembleClassifier:
    """Combina LightGBM + Random Forest por média de probabilidades
    (soft voting). Dois algoritmos com vieses diferentes tendem a
    errar em casos diferentes — a média often reduz erro que nenhum
    dos dois sozinho corrigiria. Implementa .fit/.predict como um
    estimador comum, então o resto do pipeline (salvar com joblib,
    chamar .predict) não precisa saber que é um ensemble por baixo.
    """

    def __init__(self, lgb_params: dict):
        self.lgb_params = lgb_params

    def fit(self, X, y):
        self.lgb_model = lgb.LGBMClassifier(class_weight="balanced", **self.lgb_params)
        self.lgb_model.fit(X, y)

        self.rf_model = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
        )
        self.rf_model.fit(X, y)

        self.classes_ = self.lgb_model.classes_
        return self

    def predict_proba(self, X):
        proba_lgb = self.lgb_model.predict_proba(X)
        proba_rf = self.rf_model.predict_proba(X)
        return (proba_lgb + proba_rf) / 2

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class ModelerAgent:
    # Cada retreino usa a próxima config desta lista, em vez de repetir
    # sempre a mesma — sem isso, "retreinar" era um no-op disfarçado.
    HYPERPARAM_GRID = [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": -1, "num_leaves": 31},
        {"n_estimators": 600, "learning_rate": 0.03, "max_depth": -1, "num_leaves": 31},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15},
        {"n_estimators": 800, "learning_rate": 0.02, "max_depth": 6, "num_leaves": 63},
        {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 7},
    ]

    def __init__(self, target_column: str, task: str = "regression", model_dir: str = "./models"):
        self.target_column = target_column
        self.task = task  # "regression" ou "classification"
        # classification: accuracy (maior é melhor). regression: RMSE
        # (menor é melhor). Usado pra saber se um retreino de fato ajudou.
        self.higher_is_better = task == "classification"
        self.model_dir = model_dir
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features derivadas quando as colunas-fonte existem.

        Específico do domínio (nomes de coluna do Titanic), mas o
        padrão — checar se a coluna-fonte existe antes de derivar —
        deixa seguro rodar isso em qualquer dataset sem quebrar.
        """
        df = df.copy()

        if "Name" in df.columns:
            # "Braund, Mr. Owen Harris" -> "Mr"
            df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
            title_normalize = {
                "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
                "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
                "Col": "Rare", "Don": "Rare", "Dr": "Rare", "Major": "Rare",
                "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
            }
            df["Title"] = df["Title"].replace(title_normalize)
            common_titles = {"Mr", "Miss", "Mrs", "Master"}
            df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")

        if "SibSp" in df.columns and "Parch" in df.columns:
            df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
            df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

        health_cols = {
            "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
            "step_count", "exercise_duration", "water_intake",
        }
        if health_cols.issubset(df.columns):
            # Faixas clínicas padrão de BMI — categórico, vira sinal
            # forte via one-hot (o valor numérico bruto já entra
            # como feature, isso captura o efeito de limiar).
            df["bmi_category"] = pd.cut(
                pd.to_numeric(df["bmi"], errors="coerce"),
                bins=[-np.inf, 18.5, 25, 30, np.inf],
                labels=["underweight", "normal", "overweight", "obese"],
            ).astype(str)

            # +1 no denominador evita divisão por zero sem descartar
            # linhas; olhando as unidades, isso é irrelevante no
            # resultado pra valores normais dessas colunas.
            df["calories_per_step"] = df["calorie_expenditure"] / (df["step_count"] + 1)
            df["steps_per_exercise_min"] = df["step_count"] / (df["exercise_duration"] + 1)
            df["water_per_bmi"] = df["water_intake"] / (df["bmi"] + 1)
            df["heart_rate_bmi_interaction"] = df["heart_rate"] * df["bmi"]
            # Déficit de sono: quanto abaixo de 8h, zerado se dormiu o
            # suficiente (não queremos "sono extra" cancelando déficit
            # de outro dia na mesma linha).
            df["sleep_deficit"] = (8 - df["sleep_duration"]).clip(lower=0)
            df["activity_per_calorie"] = df["exercise_duration"] / (df["calorie_expenditure"] + 1)

        return df

    def prepare_features(self, state: PipelineState) -> PipelineState:
        state.note("[modeler] preparando features")
        df = pd.read_csv(state.train_df_path)
        df = self._engineer_features(df)

        state.target_column = self.target_column
        # Colunas de identificador/texto livre nunca ajudam um modelo
        # tabular simples e podem vazar overfitting (ex: Name, Ticket,
        # Cabin têm alta cardinalidade) — excluídas explicitamente,
        # já que a lógica de cardinalidade abaixo também as filtraria,
        # mas ser explícito documenta a intenção.
        drop_cols = {self.target_column, "Name", "Ticket", "Cabin", "PassengerId"}
        feature_cols = [c for c in df.columns if c not in drop_cols]

        numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
        # Colunas categóricas de baixa cardinalidade (ex: Sex, Embarked,
        # Title) carregam sinal forte e não podem ser descartadas —
        # usamos one-hot encoding para incluí-las.
        categorical_cols = [
            c for c in df[feature_cols].select_dtypes(exclude="number").columns
            if df[c].nunique(dropna=True) <= 15
        ]

        dummies = pd.get_dummies(df[categorical_cols], columns=categorical_cols, dummy_na=True)
        state.feature_columns = numeric_cols + list(dummies.columns)
        state.categorical_source_columns = categorical_cols
        state.features_ready = True

        state.note(
            f"[modeler] {len(numeric_cols)} features numéricas + "
            f"{len(categorical_cols)} categóricas ({len(dummies.columns)} colunas após encoding)"
        )
        return state

    def _build_matrix(self, df: pd.DataFrame, state: PipelineState) -> pd.DataFrame:
        """Recria a mesma matriz de features do treino (features derivadas +
        numéricas + categóricas em one-hot), alinhando colunas que não
        aparecerem no dataframe atual (ex: categoria vista no treino,
        ausente no test set)."""
        df = self._engineer_features(df)
        categorical_cols = state.categorical_source_columns
        numeric_cols = [c for c in state.feature_columns if c in df.columns]

        dummies = pd.get_dummies(df[categorical_cols], columns=categorical_cols, dummy_na=True) \
            if categorical_cols else pd.DataFrame(index=df.index)

        X = pd.concat([df[numeric_cols], dummies], axis=1)
        # Garante mesmas colunas e mesma ordem do treino; preenche com 0
        # colunas que existiam no treino mas não neste dataframe.
        X = X.reindex(columns=state.feature_columns, fill_value=0)
        return X.fillna(-999)

    def train(self, state: PipelineState) -> PipelineState:
        params = self.HYPERPARAM_GRID[len(state.cv_history) % len(self.HYPERPARAM_GRID)]
        state.note(f"[modeler] treinando modelo com cross-validation, params={params}")
        df = pd.read_csv(state.train_df_path).dropna(subset=[state.target_column])

        X = self._build_matrix(df, state)
        y = df[state.target_column]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        model = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if self.task == "classification":
                model = _EnsembleClassifier(params)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = balanced_accuracy_score(y_val, preds)
            else:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = mean_squared_error(y_val, preds, squared=False)

            fold_scores.append(score)
            state.note(f"[modeler] fold {fold}: score={score:.4f}")

        avg_score = sum(fold_scores) / len(fold_scores)
        model_id = str(uuid.uuid4())[:8]
        model_path = Path(self.model_dir) / f"model_{model_id}.joblib"
        joblib.dump(model, model_path)

        state.current_model_id = model_id
        state.current_model_path = str(model_path)
        state.cv_score = avg_score
        state.cv_history.append({"model_id": model_id, "cv_score": avg_score, "params": params})

        improved = (
            state.best_cv_score is None
            or (self.higher_is_better and avg_score > state.best_cv_score)
            or (not self.higher_is_better and avg_score < state.best_cv_score)
        )
        if improved:
            state.best_cv_score = avg_score
            state.no_improve_count = 0
        else:
            state.no_improve_count += 1

        state.note(
            f"[modeler] modelo {model_id} treinado, cv_score médio={avg_score:.4f} "
            f"(melhor até agora={state.best_cv_score:.4f}, "
            f"sem melhora há {state.no_improve_count} tentativa(s))"
        )
        return state

    def predict(self, state: PipelineState) -> PipelineState:
        """Gera predictions.csv sobre o test set com o modelo atual."""
        state.note("[modeler] gerando predictions no test set")
        model = joblib.load(state.current_model_path)
        df_test = pd.read_csv(state.test_df_path)

        X_test = self._build_matrix(df_test, state)
        preds = model.predict(X_test)

        out_path = Path(state.current_model_path).parent / f"predictions_{state.current_model_id}.csv"
        pd.DataFrame({"prediction": preds}).to_csv(out_path, index=False)
        state.predictions_path = str(out_path)
        return state
