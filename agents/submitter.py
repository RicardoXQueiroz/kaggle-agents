"""Agente responsável por formatar e enviar submissões via Kaggle API."""
from datetime import datetime

import pandas as pd

from core.kaggle_client import KaggleClient
from core.state import PipelineState, Submission


class SubmitterAgent:
    def __init__(self, kaggle_client: KaggleClient, id_column: str = "id"):
        self.kaggle_client = kaggle_client
        self.id_column = id_column

    def run(self, state: PipelineState) -> PipelineState:
        state.note(f"[submitter] formatando submissão do modelo {state.current_model_id}")

        # Junta o id do test set original com as predictions geradas
        df_test = pd.read_csv(state.test_df_path)
        preds = pd.read_csv(state.predictions_path)

        submission = pd.DataFrame({
            self.id_column: df_test[self.id_column],
            state.target_column: preds["prediction"],
        })
        submission_path = state.predictions_path.replace("predictions_", "submission_")
        submission.to_csv(submission_path, index=False)

        message = f"model={state.current_model_id} cv={state.cv_score:.4f}"
        self.kaggle_client.submit(state.competition, submission_path, message)

        state.submission_history.append(
            Submission(
                timestamp=datetime.utcnow().isoformat(),
                model_id=state.current_model_id,
                local_cv_score=state.cv_score,
                message=message,
            )
        )
        state.note(f"[submitter] enviado: {message}")
        return state
