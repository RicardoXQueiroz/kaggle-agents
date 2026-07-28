"""Wrapper fino sobre a Kaggle API oficial.

Requer credenciais em ~/.kaggle/kaggle.json (baixadas em
https://www.kaggle.com/settings -> API -> Create New Token).

Isolar a API aqui facilita trocar por mocks em teste, sem
espalhar chamadas 'kaggle.api.xxx' pelo resto do código.
"""
from pathlib import Path


class KaggleSubmissionLimitReached(RuntimeError):
    """Levantada quando a Kaggle recusa a submissão por limite diário
    atingido — um estado esperado, não uma falha do sistema. O
    orquestrador trata isso como uma parada normal, não como erro fatal."""


class KaggleClient:
    def __init__(self):
        # O pacote kaggle mais recente autentica automaticamente no
        # 'import kaggle', lendo ~/.kaggle/access_token (ou a env var
        # KAGGLE_API_TOKEN). Chamar KaggleApi().authenticate() manualmente
        # depois disso falha, pois o token já foi consumido — por isso
        # usamos a instância já autenticada em kaggle.api.
        import kaggle

        self.api = kaggle.api

    def download_competition_files(self, competition: str, dest: str) -> str:
        Path(dest).mkdir(parents=True, exist_ok=True)
        self.api.competition_download_files(competition, path=dest, quiet=False)
        # A Kaggle API baixa um .zip único com o nome da competição
        zip_path = Path(dest) / f"{competition}.zip"
        if zip_path.exists():
            import zipfile
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest)
        return dest

    def submit(self, competition: str, file_path: str, message: str) -> None:
        import requests

        try:
            self.api.competition_submit(file_path, message, competition)
        except requests.exceptions.HTTPError as e:
            # O SDK novo não expõe a mensagem de erro da Kaggle no
            # traceback padrão — ela vem no corpo da resposta HTTP.
            body = e.response.text if e.response is not None else "(sem corpo de resposta)"

            if "daily submission allowance" in body.lower():
                raise KaggleSubmissionLimitReached(
                    f"Limite diário de submissões atingido para '{competition}'. "
                    f"Resposta da Kaggle: {body}"
                ) from e

            # Outras causas de 400 (ex: regras da competição não aceitas
            # no site) continuam como erro fatal — merecem investigação.
            raise RuntimeError(
                f"Falha ao submeter para '{competition}': {e}\n"
                f"Resposta da Kaggle: {body}"
            ) from e

    @staticmethod
    def _get(obj, *names, default="__raise__"):
        """Tenta vários nomes de atributo (snake_case novo vs camelCase
        antigo) — versões recentes do pacote kaggle renomearam vários
        campos dos objetos de resposta."""
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        if default != "__raise__":
            return default
        raise AttributeError(f"nenhum de {names} encontrado em {obj!r}")

    def get_leaderboard(self, competition: str) -> list[dict]:
        """Retorna a primeira página do leaderboard (top ~20). Suficiente
        para ver o score de corte no topo; para saber o total de
        participantes use get_team_count."""
        board = self.api.competition_leaderboard_view(competition)
        return [
            {
                "rank": i + 1,
                "team": self._get(entry, "team_name", "teamName"),
                "score": float(self._get(entry, "score")),
            }
            for i, entry in enumerate(board)
        ]

    def get_team_count(self, competition: str) -> int:
        """Número total de equipes/participantes da competição, vindo
        dos metadados (não do leaderboard, que só devolve uma página)."""
        response = self.api.competitions_list(search=competition)
        matches = self._get(response, "competitions", default=response)
        for c in matches:
            ref = self._get(c, "ref", "competitionId", default=None)
            if ref and competition in str(ref):
                return int(self._get(c, "team_count", "teamCount"))
        raise ValueError(f"Competição '{competition}' não encontrada em competitions_list")

    def get_my_submissions(self, competition: str) -> list[dict]:
        subs = self.api.competition_submissions(competition)
        return [
            {
                "fileName": self._get(s, "file_name", "fileName"),
                "date": str(self._get(s, "date")),
                "status": self._get(s, "status"),
                "publicScore": self._get(s, "public_score", "publicScore"),
            }
            for s in subs
        ]
