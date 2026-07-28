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

## Automação via GitHub Actions

O workflow em `.github/workflows/run.yml` roda o orquestrador sozinho, a
cada 6 horas, sem você precisar estar com o notebook aberto.

**Setup (uma vez só):**

1. Suba este projeto pra um repositório no GitHub (pode ser privado).
2. Vá em **Settings > Secrets and variables > Actions > New repository
   secret**, crie um secret chamado `KAGGLE_API_TOKEN` com o valor do seu
   token (o mesmo `KGAT_...` que você usa em `~/.kaggle/access_token`
   localmente).
3. Confirme que aceitou as regras da competição manualmente no site da
   Kaggle — isso a automação não pode fazer por você.
4. Pronto. O workflow já está agendado (`cron: "0 */6 * * *"`). Pra testar
   sem esperar, vá na aba **Actions** do repositório, escolha "Run Kaggle
   agent team" e clique em **Run workflow**.

**Como a persistência funciona:**

Cada execução do GitHub Actions roda numa VM nova, que não guarda nada de
uma vez pra outra. Por isso o orquestrador salva o progresso em
`state.json` a cada passo, e o workflow faz commit desse arquivo de volta
no repositório ao final. Na próxima execução, o orquestrador carrega esse
`state.json` e continua de onde parou — sem re-treinar do zero, sem perder
o histórico de submissões.

Dados baixados (`data/`) e modelos treinados (`models/`) **não** são
versionados (veja `.gitignore`) — são recriados a cada execução, o que é
rápido e evita o problema de redistribuir dados de competição num
repositório.

**Cuidado com o limite de submissões:** o `max_submissions_per_run=5`
protege uma única execução, mas se o workflow rodar várias vezes ao dia
(a cada 6h = 4x/dia) e cada uma submeter algumas vezes, ainda dá pra
esbarrar no limite diário da Kaggle. Ajuste o `cron` pra rodar com menos
frequência (ex: uma vez por dia) se isso acontecer.

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
