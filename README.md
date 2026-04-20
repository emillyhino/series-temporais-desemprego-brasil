# 📊 Séries Temporais — Previsão do Desemprego no Brasil

Análise histórica e previsão da taxa de desemprego no Brasil utilizando dados da PNAD Contínua (IBGE), com segmentação por região, gênero e faixa etária, além de modelagem com Prophet.

## 📌 O que o projeto faz

- Coleta automática de dados da API do IBGE (PNAD Contínua)
- Análise exploratória da taxa de desemprego ao longo do tempo
- Identificação de padrões sazonais e tendências
- Decomposição da série temporal (tendência, sazonalidade e ruído)
- Geração de estatísticas descritivas (mínimo, máximo, média)
- Previsão da taxa de desemprego para os próximos períodos com Prophet
- Exportação dos dados tratados para CSV

## 📈 Resultados

- Período analisado: **2012 a 2026**
- Taxa mínima: **5.1% (out-nov-dez 2025)**
- Taxa máxima: **14.9% (jul-ago-set 2020)**
- Taxa atual: **5.8%**
- Média geral: **9.7%**
- Tendência recente de queda no desemprego com projeções para os próximos anos

## 📊 Gráficos gerados

| Arquivo | Descrição |
|--------|----------|
| `eda_desemprego.png` | Evolução histórica da taxa de desemprego |
| `decomposicao_serie.png` | Decomposição da série temporal |
| `previsao_prophet.png` | Previsão futura com intervalos de confiança |

## 🧠 Modelagem

- Modelo utilizado: **Prophet (Meta)**
- Tipo: Série temporal univariada
- Frequência: Trimestral
- Saída: Previsão com limites inferior e superior (intervalo de confiança)

## 🛠️ Tecnologias

- Python 3  
- Pandas · NumPy  
- Matplotlib · Seaborn  
- Prophet (Meta)  
- Requests (API IBGE)

## ▶️ Como executar

```bash
git clone https://github.com/emillyhino/series-temporais-desemprego-brasil.git
cd series-temporais-desemprego-brasil

pip install pandas numpy matplotlib seaborn prophet requests

python analise.py
```

## 🌐 Fonte dos dados

Dados públicos da PNAD Contínua — IBGE (API oficial)  
https://servicodados.ibge.gov.br/api/docs/agregados?versao=3

## 📂 Saídas geradas

- `eda_desemprego.png`
- `decomposicao_serie.png`
- `previsao_prophet.png`
- `desemprego_historico.csv`

## 👩‍💻 Autora

**Emilly Hino**  
Bacharela em Ciência de Dados 
[LinkedIn](linkedin.com/in/emilly-h-3626b1128)  
[GitHub](https://github.com/emillyhino)
