# federated-learning-privacy-mesh

[![CI](https://github.com/FelipeMadson/federated-learning-privacy-mesh/actions/workflows/ci.yml/badge.svg)](https://github.com/FelipeMadson/federated-learning-privacy-mesh/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)](pyproject.toml)

> Mesh Orquestrador de Aprendizado Federado de Alta Performance com Privacidade Diferencial de Rényi e Agregação Segura Multi-Krum Resiliente a Ataques Bizantinos, implementado em Python 100% puro sem dependências externas.

Arquitetado e mantido por **[Felipe Madison](https://github.com/FelipeMadson)** para viabilizar IA colaborativa multi-institucional e espaços de dados de pesquisa privativa (como os Espaços de Dados da RNP, consórcios de hospitais universitários e bancos genômicos) sem compartilhamento de dados brutos e imune a ataques de inversão de modelo.

---

## 🚀 Destaques Arquiteturais

1. **Privacidade Diferencial $(\epsilon, \delta)$ com Garantia Formal**:
   - **Corte de Norma $L_2$ de Gradientes**: Limita a sensibilidade $C$ das atualizações nos nós de borda antes do tráfego em rede.
   - **Mecanismo Gaussiano Calibrado**: Perturbação com ruído gaussiano proporcional à sensibilidade ($\sigma = \text{multiplier} \cdot C$).
   - **Contador de Privacidade de Rényi**: Cálculo rigoroso de gasto cumulativo de épsilon ao longo das rodadas via Teorema de Composição Avançada.

2. **Agregação Segura Multi-Krum Resiliente a Ataques Bizantinos**:
   - Filtro geométrico tolerante a até $f$ nós atacantes ou comprometidos ($n \ge 2f + 3$).
   - Avalia distâncias euclidianas relativas e soma distâncias aos $n - f - 2$ vizinhos mais próximos, rejeitando atualizações anômalas (ataques de envenenamento de dados).
   - Suporte adicional a **FedAvg** ponderado por amostras e **Média Aparada (Trimmed Mean)** por coordenadas.

3. **Daemon Coordenador REST Autônomo**:
   - Servidor HTTP nativo sem dependências construído sobre `http.server.HTTPServer`.
   - Endpoints:
     - `GET /health` — Verificação de integridade operacional.
     - `GET /model` — Pesos atuais do modelo global e número da rodada.
     - `GET /metrics` — Contagem de nós ativos, gasto de privacidade $(\epsilon, \delta)$ e telemetria.
     - `POST /register` — Cadastro de novas instituições participantes.
     - `POST /submit_update` — Ingestão de vetores de parâmetros de borda.
     - `POST /finalize_round` — Execução da agregação segura e avanço do modelo.

4. **100% Zero Dependências de Terceiros**:
   - Utiliza exclusivamente as bibliotecas padrão do Python 3.12 (`math`, `random`, `dataclasses`, `http.server`, `json`, `unittest`).

---

## 🛠️ Verificação & Bateria de Testes

Execute a suíte de testes com o `unittest` padrão:

```bash
python -m unittest discover tests
```

### Testes Aprovados

```text
Ran 8 tests in 0.001s

OK
```

---

## 📜 Licença

Licença MIT. Arquitetado e mantido por [Felipe Madison](https://github.com/FelipeMadson).
