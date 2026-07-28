"""Agente responsável por baixar os dados da competição e gerar um
resumo de EDA (exploratory data analysis) básico e automático.
"""
from pathlib import Path

import pandas as pd

from core.kaggle_client import KaggleClient
from core.state import PipelineState


class DataCollectorAgent:
    def __init__(self, kaggle_client: KaggleClient, workdir: str = "./data"):
        self.kaggle_client = kaggle_client
        self.workdir = workdir

    def run(self, state: PipelineState) -> PipelineState:
        state.note(f"[data_collector] baixando dados de '{state.competition}'")

        dest = Path(self.workdir) / state.competition
        self.kaggle_client.download_competition_files(state.competition, str(dest))
        state.raw_data_path = str(dest)

        train_path = dest / "train.csv"
        test_path = dest / "test.csv"
        if not train_path.exists():
            raise FileNotFoundError(
                f"train.csv não encontrado em {dest}. Verifique o nome dos "
                "arquivos desta competição (nem todas usam train.csv/test.csv)."
            )

        state.train_df_path = str(train_path)
        state.test_df_path = str(test_path) if test_path.exists() else None

        state.eda_summary = self._quick_eda(train_path)
        state.note(f"[data_collector] EDA: {state.eda_summary}")
        return state

    @staticmethod
    def _quick_eda(train_path: Path) -> dict:
        df = pd.read_csv(train_path)
        return {
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "null_pct": {
                c: round(float(df[c].isna().mean()) * 100, 2) for c in df.columns
            },
        }
