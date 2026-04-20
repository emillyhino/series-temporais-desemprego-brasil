import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
warnings.filterwarnings("ignore")

from prophet import Prophet

print("=" * 60)
print(" SÉRIES TEMPORAIS — DESEMPREGO BRASIL (PNAD)")
print("=" * 60)

# ── 1. COLETA DE DADOS — API IBGE SIDRA ───────────────────────────
print("\n[1/6] Coletando dados da API do IBGE...")

url = "https://apisidra.ibge.gov.br/values/t/6381/n1/all/v/4099/p/all/d/v4099%201"
response = requests.get(url, timeout=30)
dados = response.json()

df_raw = pd.DataFrame(dados[1:], columns=dados[0].keys())
df_raw = df_raw[["D3N", "V"]].copy()
df_raw.columns = ["trimestre", "taxa_desemprego"]
df_raw["taxa_desemprego"] = pd.to_numeric(df_raw["taxa_desemprego"], errors="coerce")
df_raw = df_raw.dropna()

# Ver formato real
print("Exemplos de trimestre:", df_raw["trimestre"].head(5).tolist())

# Converter trimestre para data
meses_pt = {
    "jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,
    "jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12
}

def trimestre_para_data(t):
    partes = t.split()
    mes_str = partes[0].split("-")[0]
    ano = int(partes[-1])
    mes = meses_pt.get(mes_str, 1)
    return pd.Timestamp(year=ano, month=mes, day=1)
df_raw["data"] = df_raw["trimestre"].apply(trimestre_para_data)
df_raw = df_raw.sort_values("data").reset_index(drop=True)

print(f"Períodos coletados: {len(df_raw)}")
print(f"Período: {df_raw['data'].min().strftime('%Y-%m')} a {df_raw['data'].max().strftime('%Y-%m')}")
print(f"Taxa atual: {df_raw['taxa_desemprego'].iloc[-1]}%")

# ── 2. EDA ────────────────────────────────────────────────────────
print("\n[2/6] Análise exploratória...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Série histórica
axes[0,0].plot(df_raw["data"], df_raw["taxa_desemprego"],
               color="#D85A30", linewidth=2)
axes[0,0].fill_between(df_raw["data"], df_raw["taxa_desemprego"],
                        alpha=0.15, color="#D85A30")
axes[0,0].axhline(y=df_raw["taxa_desemprego"].mean(), color="gray",
                  linestyle="--", alpha=0.7, label=f"Média: {df_raw['taxa_desemprego'].mean():.1f}%")
axes[0,0].set_title("Taxa de Desemprego — Brasil (PNAD Contínua)", fontsize=13, fontweight="bold")
axes[0,0].set_ylabel("Taxa (%)")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Distribuição
axes[0,1].hist(df_raw["taxa_desemprego"], bins=15, color="#534AB7",
               edgecolor="white", alpha=0.8)
axes[0,1].axvline(df_raw["taxa_desemprego"].mean(), color="#D85A30",
                  linestyle="--", label=f"Média: {df_raw['taxa_desemprego'].mean():.1f}%")
axes[0,1].set_title("Distribuição da Taxa de Desemprego")
axes[0,1].set_xlabel("Taxa (%)")
axes[0,1].legend()

# Média por ano
df_raw["ano"] = df_raw["data"].dt.year
media_ano = df_raw.groupby("ano")["taxa_desemprego"].mean()
cores = ["#D85A30" if v > df_raw["taxa_desemprego"].mean() else "#1D9E75"
         for v in media_ano]
axes[1,0].bar(media_ano.index, media_ano.values, color=cores, edgecolor="white")
axes[1,0].axhline(y=df_raw["taxa_desemprego"].mean(), color="gray",
                  linestyle="--", alpha=0.7)
axes[1,0].set_title("Média anual da taxa de desemprego")
axes[1,0].set_xlabel("Ano")
axes[1,0].set_ylabel("Taxa (%)")

# Variação trimestral
df_raw["variacao"] = df_raw["taxa_desemprego"].diff()
cores_var = ["#1D9E75" if v <= 0 else "#D85A30" for v in df_raw["variacao"].fillna(0)]
axes[1,1].bar(df_raw["data"], df_raw["variacao"].fillna(0),
              color=cores_var, edgecolor="white", width=60)
axes[1,1].axhline(y=0, color="black", linewidth=0.8)
axes[1,1].set_title("Variação trimestral do desemprego")
axes[1,1].set_ylabel("Variação (p.p.)")

plt.tight_layout()
plt.savefig("eda_desemprego.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico EDA salvo.")

# ── 3. DECOMPOSIÇÃO DA SÉRIE ──────────────────────────────────────
print("\n[3/6] Decomposição da série temporal...")

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

ts = df_raw.set_index("data")["taxa_desemprego"]
decomp = seasonal_decompose(ts, model="additive", period=4)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
decomp.observed.plot(ax=axes[0], color="#D85A30")
axes[0].set_title("Série original")
decomp.trend.plot(ax=axes[1], color="#534AB7")
axes[1].set_title("Tendência")
decomp.seasonal.plot(ax=axes[2], color="#1D9E75")
axes[2].set_title("Sazonalidade")
decomp.resid.plot(ax=axes[3], color="#888780")
axes[3].set_title("Resíduo")

for ax in axes:
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)

plt.suptitle("Decomposição da Série — Taxa de Desemprego Brasil",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("decomposicao_serie.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico de decomposição salvo.")

# ── 4. PREVISÃO COM PROPHET ───────────────────────────────────────
print("\n[4/6] Treinando modelo Prophet...")

df_prophet = df_raw[["data", "taxa_desemprego"]].rename(
    columns={"data": "ds", "taxa_desemprego": "y"}
)

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="additive",
    changepoint_prior_scale=0.05
)
model.fit(df_prophet)

# Prever 8 trimestres (2 anos)
futuro = model.make_future_dataframe(periods=8, freq="QS")
previsao = model.predict(futuro)

print(f"Previsão para os próximos 2 anos gerada!")
print(f"\nPróximos trimestres previstos:")
prev_futuro = previsao[previsao["ds"] > df_raw["data"].max()][["ds","yhat","yhat_lower","yhat_upper"]]
prev_futuro.columns = ["data", "previsao", "limite_inferior", "limite_superior"]
prev_futuro["previsao"] = prev_futuro["previsao"].round(1)
prev_futuro["limite_inferior"] = prev_futuro["limite_inferior"].round(1)
prev_futuro["limite_superior"] = prev_futuro["limite_superior"].round(1)
print(prev_futuro.to_string(index=False))

# ── 5. GRÁFICO DA PREVISÃO ────────────────────────────────────────
print("\n[5/6] Gerando gráfico de previsão...")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df_prophet["ds"], df_prophet["y"],
        color="#D85A30", linewidth=2, label="Dados reais", zorder=5)
ax.plot(previsao["ds"], previsao["yhat"],
        color="#534AB7", linewidth=2, linestyle="--", label="Previsão Prophet")
ax.fill_between(previsao["ds"], previsao["yhat_lower"], previsao["yhat_upper"],
                alpha=0.15, color="#534AB7", label="Intervalo de confiança")
ax.axvline(x=df_raw["data"].max(), color="gray",
           linestyle="--", alpha=0.7, label="Início da previsão")
ax.set_title("Previsão da Taxa de Desemprego — Brasil (Prophet)", fontsize=14, fontweight="bold")
ax.set_ylabel("Taxa de Desemprego (%)")
ax.set_xlabel("Ano")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("previsao_prophet.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico de previsão salvo.")

# ── 6. ESTATÍSTICAS FINAIS ────────────────────────────────────────
print("\n[6/6] Estatísticas finais...")

print(f"\n{'='*40}")
print(f" RESUMO DA ANÁLISE")
print(f"{'='*40}")
print(f" Período analisado: {df_raw['data'].min().year} — {df_raw['data'].max().year}")
print(f" Taxa mínima:  {df_raw['taxa_desemprego'].min():.1f}% ({df_raw.loc[df_raw['taxa_desemprego'].idxmin(), 'trimestre']})")
print(f" Taxa máxima:  {df_raw['taxa_desemprego'].max():.1f}% ({df_raw.loc[df_raw['taxa_desemprego'].idxmax(), 'trimestre']})")
print(f" Taxa atual:   {df_raw['taxa_desemprego'].iloc[-1]:.1f}%")
print(f" Média geral:  {df_raw['taxa_desemprego'].mean():.1f}%")
print(f"{'='*40}")

df_raw.to_csv("desemprego_historico.csv", index=False, encoding="utf-8-sig")
print("\n✅ Análise completa! Arquivos gerados:")
print("  - eda_desemprego.png")
print("  - decomposicao_serie.png")
print("  - previsao_prophet.png")
print("  - desemprego_historico.csv")
