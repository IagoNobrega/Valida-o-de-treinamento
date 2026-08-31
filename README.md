# YOLO Training Analytics

MVP em Streamlit para validar um modelo YOLO de referência e um candidato com o mesmo dataset e a mesma configuração, armazenando resultados reproduzíveis.

## Instalação

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uso

Na aba **Nova comparação**, informe ou envie as pastas finais do treinamento antigo e do novo. A aplicação localiza automaticamente o arquivo `weights/best.pt` de cada pasta e registra os artefatos encontrados, normalmente:

```text
meu-treino/
├── weights/
│   ├── best.pt
│   └── last.pt
├── args.yaml
└── results.csv
```

Também informe o `data.yaml` de validação, split e parâmetros. Se as duas pastas contiverem exatamente o mesmo `data.yaml`, ele é usado automaticamente. As imagens do split devem estar acessíveis no computador que executa o Streamlit.

A pasta de treinamento traz o histórico daquele treino, mas não substitui a validação comparativa: o sistema sempre executa os dois `best.pt` com uma única `ValidationConfig` imutável. Isso impede comparar métricas históricas geradas sobre datasets ou parâmetros diferentes.

Os resultados ficam em `results/CMP-.../`: `comparison.json`, CSVs, subpastas da execução YOLO e `report.md`. O `comparison.json` inclui os caminhos, argumentos e resumo de métricas finais encontrados nas duas pastas de treino. Pastas enviadas pelo navegador são copiadas temporariamente para `uploads/`, que é ignorada pelo Git. O histórico fica em `database/analytics.db`; o log técnico em `logs/app.log`.

## Estrutura

- `core/`: validação, execução YOLO, extração de métricas e regras de comparação.
- `database/`: SQLite.
- `reports/`: relatório Markdown.
- `tests/`: testes unitários sem GPU.

## Testes

```bash
pytest -q
```

Métricas por classe só são persistidas quando a versão instalada do Ultralytics as expõe na resposta de validação; o sistema não cria valores substitutos.
