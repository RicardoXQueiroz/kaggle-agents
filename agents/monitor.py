"""Agente responsável por checar o leaderboard e avaliar se a meta
(top X%) foi atingida."""
from core.kaggle_client import KaggleClient
from core.state import PipelineState


class MonitorAgent:
    def __init__(self, kaggle_client: KaggleClient):
        self.kaggle_client = kaggle_client

    def check_leaderboard(self, state: PipelineState) -> PipelineState:
        state.note("[monitor] consultando leaderboard")
        board = self.kaggle_client.get_leaderboard(state.competition)
        my_subs = self.kaggle_client.get_my_submissions(state.competition)

        state.leaderboard_total = self.kaggle_client.get_team_count(state.competition)

        def has_score(s):
            v = s["publicScore"]
            return v is not None and str(v).strip() != ""

        best_sub = max(
            (s for s in my_subs if has_score(s)),
            key=lambda s: s["date"],
            default=None,
        )
        if best_sub:
            state.best_public_score = float(best_sub["publicScore"])
            # board só traz a primeira página (top ~20 scores). Se nosso
            # score for pior que todos ali, só sabemos que a posição é
            # "pior que len(board)" — não a posição exata.
            better = sum(1 for b in board if b["score"] > state.best_public_score)
            if better < len(board):
                state.leaderboard_position = better + 1
                state.note(
                    f"[monitor] posição {state.leaderboard_position}/{state.leaderboard_total}"
                )
            else:
                state.leaderboard_position = None
                state.note(
                    f"[monitor] score {state.best_public_score} fora do top "
                    f"{len(board)} — posição exata desconhecida, mas certamente "
                    f"não está no topo do ranking ainda"
                )
        else:
            state.note(
                "[monitor] nenhuma submissão com score público ainda "
                "(a Kaggle pode levar alguns minutos para pontuar arquivos "
                "grandes) — tentaremos de novo na próxima execução"
            )
        return state

    def goal_reached(self, state: PipelineState) -> bool:
        if not state.leaderboard_position or not state.leaderboard_total:
            return False
        percentile = state.leaderboard_position / state.leaderboard_total
        return percentile <= state.target_percentile
