import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon
import scikit_posthocs as sp
from pymfe.mfe import MFE

warnings.filterwarnings('ignore')
np.random.seed(42)

print("1. Gerando dados sintéticos (35 datasets) e extraindo PyMFE...")
matriz_x_base = []
complexidades = []
datasets = []

# Configuramos o MFE para a família de complexidade buscando as médias
mfe = MFE(groups=["complexity"], summary=["mean"])

for i in range(35):
    n_samples = np.random.randint(200, 1000)
    n_features = np.random.randint(10, 50)
    n_classes = np.random.randint(2, 5)
    n_informative = np.random.randint(5, n_features)
    
    X_data, y_data = make_classification(
        n_samples=n_samples, n_features=n_features, 
        n_informative=n_informative, n_redundant=0, 
        n_classes=n_classes, random_state=i
    )
    
    nome_ds = f'DS_{i}'
    datasets.append((nome_ds, X_data, y_data))
    
    # 1.1 Extração da Base
    matriz_x_base.append({
        'Dataset': nome_ds, 
        'NumberOfInstances': n_samples, 
        'NumberOfFeatures': n_features, 
        'NumberOfClasses': n_classes
    })
    
    # 1.2 Extração da Complexidade (Instance Hardness / PyMFE)
    mfe.fit(X_data, y_data)
    ft, vals = mfe.extract()
    complexidades.append(dict(zip(ft, vals)))

df_x_base = pd.DataFrame(matriz_x_base).set_index('Dataset')
df_comp = pd.DataFrame(complexidades, index=df_x_base.index)

# Selecionamos as aproximações das métricas da nossa Tabela (kDN, TD, CL, etc.)
colunas_alvo = [c for c in df_comp.columns if any(x in c.lower() for x in ['kdn', 'minority', 'tree_depth', 'cls', 't1', 't2', 'f1', 'f2'])]
df_comp_filtrado = df_comp[colunas_alvo].replace([np.inf, -np.inf], np.nan)

# Tratamento rigoroso de nulos para não quebrar o modelo
df_comp_filtrado = df_comp_filtrado.dropna(axis=1, how='all')
imputer = SimpleImputer(strategy='median')
df_comp_imputado = pd.DataFrame(imputer.fit_transform(df_comp_filtrado), columns=df_comp_filtrado.columns, index=df_x_base.index)
df_comp_imputado = df_comp_imputado.loc[:, df_comp_imputado.nunique() > 1] # Remove colunas constantes

# Criação do X_estendido
df_x_ext = pd.concat([df_x_base, df_comp_imputado], axis=1)

print("2. Treinando algoritmos e gerando P e R...")
modelos = {'RF': RandomForestClassifier(n_estimators=30, random_state=42), 
           'DT': DecisionTreeClassifier(random_state=42), 
           'KNN': KNeighborsClassifier(), 
           'LogReg': LogisticRegression(max_iter=200), 
           'NB': GaussianNB()}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
resultados_folds = {algo: {} for algo in modelos.keys()}
matriz_p_dict = {algo: [] for algo in modelos.keys()}

for nome_ds, X_data, y_data in datasets:
    for nome_modelo, clf in modelos.items():
        scores = cross_val_score(clf, X_data, y_data, cv=cv, scoring='accuracy', n_jobs=-1)
        resultados_folds[nome_modelo][nome_ds] = scores
        matriz_p_dict[nome_modelo].append(np.mean(scores))

df_p = pd.DataFrame(matriz_p_dict, index=df_x_base.index)
df_r = df_p.rank(axis=1, method='average', ascending=False)
algoritmos = list(modelos.keys())
n_algos = len(algoritmos)

print("3. Avaliação LODO (Spearman, Curva de Perda e AUC)...")
# Para o gráfico não ficar ilegível, testaremos AR, MR, Wilcoxon e as versões Base vs Extendido dos regressores principais
abordagens = [
    'AR', 'MR', 'Wilcoxon', 
    'App4_Base', 'App4_Ext', 
    'App5_Base', 'App5_Ext', 
    'HARRIS_Base', 'HARRIS_Ext'
]

resultados_spearman = {ab: [] for ab in abordagens}
recomendacoes = {ab: [] for ab in abordagens}

for teste_ds in df_x_base.index:
    # Separação Base
    treino_x_base = df_x_base.drop(index=teste_ds)
    teste_x_base = df_x_base.loc[[teste_ds]]
    
    # Separação Estendida
    treino_x_ext = df_x_ext.drop(index=teste_ds)
    teste_x_ext = df_x_ext.loc[[teste_ds]]
    
    treino_p = df_p.drop(index=teste_ds)
    treino_r = df_r.drop(index=teste_ds)
    rank_verdadeiro = df_r.loc[teste_ds]

    # 1. Agregações (Não usam features)
    rank_ar = treino_r.mean(axis=0).rank(method='average')
    rank_mr = treino_r.median(axis=0).rank(method='average')

    vitorias = np.zeros(n_algos)
    for i, a1 in enumerate(algoritmos):
        for j, a2 in enumerate(algoritmos):
            if i != j:
                f1 = resultados_folds[a1][teste_ds]
                f2 = resultados_folds[a2][teste_ds]
                if not np.all(f1 == f2):
                    stat, p_val = wilcoxon(f1, f2)
                    if p_val < 0.05 and np.mean(f1) > np.mean(f2):
                        vitorias[i] += 1
    rank_wilcoxon = pd.Series(vitorias, index=algoritmos).rank(ascending=False, method='average')

    # 2. Regressores (Base vs Estendido)
    # App 4
    mod4_base = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42)).fit(treino_x_base, treino_p.values)
    rank_4_base = pd.Series(mod4_base.predict(teste_x_base)[0], index=algoritmos).rank(ascending=False)
    
    mod4_ext = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42)).fit(treino_x_ext, treino_p.values)
    rank_4_ext = pd.Series(mod4_ext.predict(teste_x_ext)[0], index=algoritmos).rank(ascending=False)

    # App 5
    mod5_base = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42)).fit(treino_x_base, treino_r.values)
    rank_5_base = pd.Series(mod5_base.predict(teste_x_base)[0], index=algoritmos).rank(ascending=True)
    
    mod5_ext = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42)).fit(treino_x_ext, treino_r.values)
    rank_5_ext = pd.Series(mod5_ext.predict(teste_x_ext)[0], index=algoritmos).rank(ascending=True)

    # HARRIS (usando lambda 0.5 como representante para simplificar)
    l = 0.5
    target_combinado = pd.concat([treino_p * np.sqrt(1 - l), treino_r * np.sqrt(l)], axis=1)
    
    mod6_base = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42)).fit(treino_x_base, target_combinado.values)
    rank_h_base = pd.Series(mod6_base.predict(teste_x_base)[0][:n_algos] / np.sqrt(1 - l), index=algoritmos).rank(ascending=False)
    
    mod6_ext = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42)).fit(treino_x_ext, target_combinado.values)
    rank_h_ext = pd.Series(mod6_ext.predict(teste_x_ext)[0][:n_algos] / np.sqrt(1 - l), index=algoritmos).rank(ascending=False)

    preds_dict = {
        'AR': rank_ar, 'MR': rank_mr, 'Wilcoxon': rank_wilcoxon, 
        'App4_Base': rank_4_base, 'App4_Ext': rank_4_ext, 
        'App5_Base': rank_5_base, 'App5_Ext': rank_5_ext,
        'HARRIS_Base': rank_h_base, 'HARRIS_Ext': rank_h_ext
    }

    # Avaliação de Spearman
    for ab in abordagens:
        rank_pred = preds_dict[ab]
        corr = spearmanr(rank_pred.values, rank_verdadeiro.values)[0]
        resultados_spearman[ab].append(corr if not np.isnan(corr) else 0)
        ordem_rec = rank_pred.sort_values().index.tolist()
        recomendacoes[ab].append(ordem_rec)

print("\n[ Tabela Final: Spearman e AUC da Curva de Perda ]")
df_spearman = pd.DataFrame(resultados_spearman).fillna(0).mean().sort_values(ascending=False)
resultados_finais = []

for ab in df_spearman.index:
    perdas_t_totais = np.zeros(n_algos)
    for idx, teste_ds in enumerate(df_x_base.index):
        max_possivel = df_p.loc[teste_ds].max()
        ordem = recomendacoes[ab][idx]
        perdas_t = []
        for t in range(1, n_algos + 1):
            melhor_testado = max([df_p.loc[teste_ds, a] for a in ordem[:t]])
            perdas_t.append(max_possivel - melhor_testado)
        perdas_t_totais += np.array(perdas_t)
    
    auc = np.mean(perdas_t_totais / len(df_x_base))
    resultados_finais.append({'Abordagem': ab, 'Spearman': df_spearman[ab], 'AUC (Perda)': auc})

print(pd.DataFrame(resultados_finais).set_index('Abordagem').round(4))

print("\n--- GERANDO GRÁFICO DE DIFERENÇA CRÍTICA COMPARATIVO ---")
rankings_df = pd.DataFrame(resultados_spearman).fillna(0)
rank_medio = rankings_df.rank(axis=1, ascending=False).mean().sort_values()

plt.figure(figsize=(10, 5))
cores = ['skyblue' if 'Base' in x or x in ['AR', 'MR', 'Wilcoxon'] else 'salmon' for x in rank_medio.index]
plt.barh(rank_medio.index, rank_medio.values, color=cores, edgecolor='black')
plt.xlabel('Rank Médio de Spearman (Menor é Melhor)', fontsize=11)
plt.title('Diferença Crítica: Dataset Base (Azul) vs Estendido (Vermelho)', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis() 
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig("diferenca_critica_comparativo.png", dpi=300, bbox_inches='tight')
print("Gráfico de Diferença Crítica Comparativo salvo com sucesso!")

import seaborn as sns

print("\n--- ETAPA 3: ANÁLISE DE REDUNDÂNCIA (CORRELAÇÃO) ---")
# Calcula a matriz de correlação de Spearman para as meta-features
matriz_corr = df_x_ext.corr(method='spearman')

plt.figure(figsize=(12, 10))
# Máscara para esconder o triângulo superior e facilitar a leitura
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
sns.heatmap(matriz_corr, mask=mask, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Mapa de Calor de Redundância: Meta-características (X_estendido)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("matriz_redundancia.png", dpi=300)
print("Gráfico de redundância gerado e guardado como 'matriz_redundancia.png'.")

print("\n--- ETAPA 4: IMPORTÂNCIA DAS META-FEATURES POR ALGORITMO BASE ---")
importancias_por_algo = {}

# Treina um Regressor (RF) para cada algoritmo base isoladamente e extrai a importância
for algoritmo in df_p.columns:
    alvo = df_p[algoritmo]
    rf_meta = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_meta.fit(df_x_ext, alvo)
    
    # Armazena as importâncias associadas ao nome das colunas
    importancias = pd.Series(rf_meta.feature_importances_, index=df_x_ext.columns)
    importancias_por_algo[algoritmo] = importancias.sort_values(ascending=False).head(5) # Top 5

# Criação de um gráfico de barras lado a lado para comparar os perfis de importância
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharex=True)
fig.suptitle('Top 5 Meta-características Mais Importantes por Classificador Base', fontsize=16, fontweight='bold', y=1.05)

for idx, (algo, imp) in enumerate(importancias_por_algo.items()):
    sns.barplot(x=imp.values, y=imp.index, ax=axes[idx], palette='viridis')
    axes[idx].set_title(algo)
    axes[idx].set_xlabel('Importância Relativa')
    if idx > 0:
        axes[idx].set_ylabel('')

plt.tight_layout()
plt.savefig("importancia_por_algoritmo.png", dpi=300, bbox_inches='tight')
print("Gráfico de importâncias gerado e guardado como 'importancia_por_algoritmo.png'.")