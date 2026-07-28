# Kaggle multi-agent team

Time de agentes que roda o ciclo completo de uma competição Kaggle:
coleta de dados → treino → submissão → monitoramento de leaderboard,
orquestrado por um loop de decisão determinístico.

## Setup

```bash
pip install -r requirements.txt
```

Credenciais da Kaggle API: baixe `kaggle.json` em
https://www.kaggle.com/settings > API > Create New Token, e coloque em
`~/.kaggle/kaggle.json` (permissão 600).

Aceite as regras da competição no site antes de rodar — a API rejeita
submissões para competições cujas regras você não aceitou manualmente.

## Rodar

```bash
python orchestrator.py
```

O exemplo em `orchestrator.py` está configurado para a competição
`titanic` (classificação, coluna alvo `Survived`, id `PassengerId`) —
boa para validar o pipeline de ponta a ponta antes de apontar para
uma competição de verdade.

## Estrutura

```
core/
  state.py           # estado compartilhado (dataclass)
  kaggle_client.py    # wrapper da Kaggle API
agents/
  data_collector.py   # baixa dados + EDA automático
  modeler.py           # feature prep + treino + CV + predict
  submitter.py         # formata e envia submissão
  monitor.py            # consulta leaderboard, avalia meta
orchestrator.py         # loop de decisão central
```

## Limitações conhecidas (próximos passos)

- **Modeler**: usa só features numéricas no baseline — falta encoding
  categórico, imputação mais robusta, feature engineering.
- **`_should_retrain`**: sempre retorna `False` no primeiro treino —
  a lógica de "quando vale a pena retreinar" ainda é ingênua.
- **Formato de submissão**: assume `test.csv` já ter uma coluna de id
  igual à do `sample_submission.csv`. Competições com formatos mais
  exóticos exigem ajuste manual em `submitter.py`.
- **Sem retry/backoff** nas chamadas à Kaggle API — a API tem rate
  limits (ex: 5 submissões/dia em algumas competições).
- **Hospedagem**: por enquanto roda local. Para automação contínua,
  próximos passos incluem empacotar em container e agendar via GitHub
  Actions (cron) ou uma VM simples — cobrimos isso quando chegarmos lá.
