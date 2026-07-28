"""Estado compartilhado entre todos os agentes.

Nenhum agente conversa diretamente com outro — todos leem e escrevem
neste objeto através do orquestrador. Isso mantém baixo acoplamento
e dá rastreabilidade total do que aconteceu em cada rodada.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import dataclasses
import json


@dataclass
class Submission:
    timestamp: str
    model_id: str
    local_cv_score: float
    public_score: Optional[float] = None
    message: str = ""


@dataclass
class PipelineState:
    competition: str

    # Data collector
    raw_data_path: Optional[str] = None
    train_df_path: Optional[str] = None
    test_df_path: Optional[str] = None
    eda_summary: dict[str, Any] = field(default_factory=dict)

    # Modeler
    features_ready: bool = False
    feature_columns: list[str] = field(default_factory=list)
    categorical_source_columns: list[str] = field(default_factory=list)
    target_column: Optional[str] = None
    current_model_id: Optional[str] = None
    current_model_path: Optional[str] = None
    cv_score: Optional[float] = None
    cv_history: list[dict] = field(default_factory=list)
    best_cv_score: Optional[float] = None
    no_improve_count: int = 0
    retrain_patience: int = 3

    # Submitter
    submission_history: list[Submission] = field(default_factory=list)
    predictions_path: Optional[str] = None

    # Monitor
    leaderboard_position: Optional[int] = None
    leaderboard_total: Optional[int] = None
    best_public_score: Optional[float] = None
    target_percentile: float = 0.20  # top 20%

    # Controle do loop
    iteration: int = 0
    max_iterations: int = 10
    max_submissions_per_run: int = 5
    submissions_this_run: int = 0
    log: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        stamp = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{stamp}] {msg}"
        self.log.append(entry)
        print(entry)


def save_state(state: PipelineState, path: str) -> None:
    """Serializa o estado pra JSON, pra sobreviver entre execuções
    (ex: entre uma rodada e outra do GitHub Actions, que roda em uma
    VM nova a cada vez — nada em disco sobrevive exceto o que está
    no repositório)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(state), f, indent=2, ensure_ascii=False)


def load_state(path: str) -> PipelineState:
    """Reconstrói o PipelineState a partir do JSON salvo por save_state."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    submissions_raw = data.pop("submission_history", [])
    state = PipelineState(**data)
    state.submission_history = [Submission(**s) for s in submissions_raw]
    return state
