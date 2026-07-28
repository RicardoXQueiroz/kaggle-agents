"""Orquestrador central: mantém o estado e decide, a cada iteração,
qual agente deve agir. Determinístico de propósito — previsível e
fácil de debugar. Pontos de "agência" via LLM podem ser injetados
depois dentro de cada agente (ex: modeler decidindo que features
criar), sem mudar este loop.
"""
import os

from agents.data_collector import DataCollectorAgent
from agents.modeler import ModelerAgent
from agents.monitor import MonitorAgent
from agents.submitter import SubmitterAgent
from core.kaggle_client import KaggleClient, KaggleSubmissionLimitReached
from core.state import PipelineState, load_state, save_state


class Orchestrator:
    def __init__(self, competition: str, target_column: str, task: str = "regression",
                 id_column: str = "id", target_percentile: float = 0.20,
                 max_iterations: int = 10, state_path: str | None = None):
        self.state_path = state_path
        loaded = None

        if state_path and os.path.exists(state_path):
            loaded = load_state(state_path)
            if loaded.competition != competition:
                # Estado salvo é de outra competição — começar do zero
                # em vez de misturar features/modelo de um dataset
                # diferente com a competição atual.
                loaded = None

        if loaded is not None:
            self.state = loaded
            self._reconcile_after_load(self.state, max_iterations)
        else:
            self.state = PipelineState(
                competition=competition,
                target_column=target_column,
                target_percentile=target_percentile,
                max_iterations=max_iterations,
            )

        kaggle_client = KaggleClient()
        self.data_collector = DataCollectorAgent(kaggle_client)
        self.modeler = ModelerAgent(target_column=target_column, task=task)
        self.submitter = SubmitterAgent(kaggle_client, id_column=id_column)
        self.monitor = MonitorAgent(kaggle_client)

    @staticmethod
    def _reconcile_after_load(state: PipelineState, max_iterations: int) -> None:
        """Ajusta o estado carregado do disco para uma execução nova.

        Cada execução do GitHub Actions roda numa VM efêmera — dados
        baixados e modelos treinados em execuções anteriores não
        existem mais no disco, mesmo que o state.json "lembre" deles.
        Sem essa reconciliação, o orquestrador tentaria carregar um
        arquivo de modelo que não existe e quebraria.
        """
        state.iteration = 0
        state.max_iterations = max_iterations
        state.no_improve_count = 0  # dá um novo "fôlego" de tentativas a cada dia
        state.submissions_this_run = 0  # o limite é por execução, não histórico total

        if state.raw_data_path and not os.path.exists(state.raw_data_path):
            state.raw_data_path = None
            state.features_ready = False

        if state.current_model_path and not os.path.exists(state.current_model_path):
            state.current_model_id = None
            state.current_model_path = None
            state.predictions_path = None

    def run(self) -> PipelineState:
        state = self.state

        while state.iteration < state.max_iterations:
            state.iteration += 1
            state.note(f"--- iteração {state.iteration} ---")

            if state.raw_data_path is None:
                state = self.data_collector.run(state)
                self._checkpoint(state)
                continue

            if not state.features_ready:
                state = self.modeler.prepare_features(state)
                self._checkpoint(state)
                continue

            if state.current_model_id is None or self._should_retrain(state):
                state = self.modeler.train(state)
                self._checkpoint(state)
                continue

            if state.predictions_path is None or self._is_new_model_unsubmitted(state):
                if state.submissions_this_run >= state.max_submissions_per_run:
                    state.note(
                        f"[orchestrator] limite de {state.max_submissions_per_run} "
                        "submissões desta execução atingido. Parando para não "
                        "esbarrar no limite diário da Kaggle."
                    )
                    break
                state = self.modeler.predict(state)
                try:
                    state = self.submitter.run(state)
                except KaggleSubmissionLimitReached as e:
                    state.note(
                        f"[orchestrator] {e} — isso é esperado, não um bug. "
                        "Parando aqui; a próxima execução agendada continua "
                        "de onde este estado ficou salvo."
                    )
                    self._checkpoint(state)
                    break
                state.submissions_this_run += 1
                self._checkpoint(state)
                continue

            state = self.monitor.check_leaderboard(state)
            if self.monitor.goal_reached(state):
                state.note("[orchestrator] meta atingida (top "
                           f"{int(state.target_percentile * 100)}%). Encerrando.")
                self._checkpoint(state)
                break

            if state.no_improve_count >= state.retrain_patience:
                state.note(
                    f"[orchestrator] {state.no_improve_count} tentativas sem melhora "
                    f"(paciência={state.retrain_patience}). Parando — modelo atual "
                    f"provavelmente já é o melhor que essa abordagem consegue."
                )
                self._checkpoint(state)
                break

            state.note("[orchestrator] meta não atingida, retreinando com ajuste")
            self._trigger_retrain(state)
            self._checkpoint(state)

        else:
            state.note("[orchestrator] limite de iterações atingido sem alcançar a meta.")
            self._checkpoint(state)

        return state

    def _checkpoint(self, state: PipelineState) -> None:
        """Salva o estado em disco após cada passo, não só no final —
        se a execução for interrompida (ex: timeout do GitHub Actions),
        o progresso até ali não se perde."""
        if self.state_path:
            save_state(state, self.state_path)

    @staticmethod
    def _should_retrain(state: PipelineState) -> bool:
        # current_model_id é resetado por _trigger_retrain quando o
        # monitor decide que vale tentar de novo — isso já basta pra
        # cair na condição "current_model_id is None" no loop principal.
        return False

    @staticmethod
    def _is_new_model_unsubmitted(state: PipelineState) -> bool:
        submitted_ids = {s.model_id for s in state.submission_history}
        return state.current_model_id not in submitted_ids

    @staticmethod
    def _trigger_retrain(state: PipelineState) -> None:
        # Força uma nova rodada de treino no próximo loop.
        state.current_model_id = None
        state.predictions_path = None


if __name__ == "__main__":
    orch = Orchestrator(
        competition=os.environ.get("KAGGLE_COMPETITION", "titanic"),
        target_column=os.environ.get("TARGET_COLUMN", "Survived"),
        task=os.environ.get("TASK", "classification"),
        id_column=os.environ.get("ID_COLUMN", "PassengerId"),
        target_percentile=float(os.environ.get("TARGET_PERCENTILE", "0.20")),
        # 4 iterações pro ciclo inicial (coleta, features, treino,
        # predict+submit) + 3 iterações por rodada de retreino
        # (treino, predict+submit, monitor). Com retrain_patience=3,
        # o pior caso é ~4 + 3*4 = 16; deixamos folga.
        max_iterations=int(os.environ.get("MAX_ITERATIONS", "25")),
        state_path=os.environ.get("STATE_PATH", "state.json"),
    )
    final_state = orch.run()
    print("\n=== resumo final ===")
    print(f"cv_score (última tentativa): {final_state.cv_score}")
    print(f"melhor cv_score: {final_state.best_cv_score}")
    print(f"tentativas de treino: {len(final_state.cv_history)}")
    print(f"leaderboard: {final_state.leaderboard_position}/{final_state.leaderboard_total}")
    print(f"estado salvo em: {orch.state_path}")
