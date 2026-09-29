import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr, wilcoxon
from pymfe.mfe import MFE

warnings.filterwarnings('ignore')
np.random.seed(42)

print("1. Gerando dados sintéticos (35 datasets) e extraindo PyMFE...")
matriz_x_base = []
complexidades = []
datasets = []

mfe = MFE(groups=["complexity"], summary=["mean"])
tempo_total_extracao = 0

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
    
    matriz_x_base.append({
        'Dataset': nome_ds, 
        'NumberOfInstances': n_samples, 
        'NumberOfFeatures': n_features, 
        'NumberOfClasses': n_classes
    })
    
    inicio_extracao = time.time()
    mfe.fit(X_data, y_data)
    ft, vals = mfe.extract()
    fim_extracao = time.time()
    
    tempo_total_extracao += (fim_extracao - inicio_extracao)
    complexidades.append(dict(zip(ft, vals)))

print(f"\n[!] Tempo total de extração PyMFE (35 datasets): {tempo_total_extracao:.2f} segundos\n")

df_x_base = pd.DataFrame(matriz_x_base).set_index('Dataset')
df_comp = pd.DataFrame(complexidades, index=df_x_base.index)

colunas_alvo = [c for c in df_comp.columns if any(x in c.lower() for x in ['kdn', 'minority', 'tree_depth', 'cls', 't1', 't2', 'f1', 'f2'])]
df_comp_filtrado = df_comp[colunas_alvo].replace([np.inf, -np.inf], np.nan)
df_comp_filtrado = df_comp_filtrado.dropna(axis=1, how='all')

imputer = SimpleImputer(strategy='median')
df_comp_imputado = pd.DataFrame(imputer.fit_transform(df_comp_filtrado), columns=df_comp_filtrado.columns, index=df_x_base.index)
df_comp_imputado = df_comp_imputado.loc[:, df_comp_imputado.nunique() > 1] 

df_x_ext = pd.concat([df_x_base, df_comp_imputado], axis=1)

print(f"[!] Dimensões de X_base: {df_x_base.shape}")
print(f"[!] Dimensões de X_estendido: {df_x_ext.shape}\n")

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
abordagens = [
    'AR', 'MR', 'Wilcoxon', 
    'App4_Base', 'App4_Ext', 
    'App5_Base', 'App5_Ext', 
    'HARRIS_Base', 'HARRIS_Ext'
]

resultados_spearman = {ab: [] for ab in abordagens}
recomendacoes = {ab: [] for ab in abordagens}

for teste_ds in df_x_base.index:
    treino_x_base = df_x_base.drop(index=teste_ds)
    teste_x_base = df_x_base.loc[[teste_ds]]
    
    treino_x_ext = df_x_ext.drop(index=teste_ds)
    teste_x_ext = df_x_ext.loc[[teste_ds]]
    
    treino_p = df_p.drop(index=teste_ds)
    treino_r = df_r.drop(index=teste_ds)
    rank_verdadeiro = df_r.loc[teste_ds]

    rank_ar = treino_r.mean(axis=0).rank(method='average')
    rank_mr = treino_r.median(axis=0).rank(method='average')

    vitorias = np.zeros(n_algos)
    for i, a1 in enumerate(algoritmos):
        for j, a2 in enumerate(algoritmos):
            if i != j:
                f1_scores = resultados_folds[a1][teste_ds]
                f2_scores = resultados_folds[a2][teste_ds]
                if not np.all(f1_scores == f2_scores):
                    stat, p_val = wilcoxon(f1_scores, f2_scores)
                    if p_val < 0.05 and np.mean(f1_scores) > np.mean(f2_scores):
                        vitorias[i] += 1
    rank_wilcoxon = pd.Series(vitorias, index=algoritmos).rank(ascending=False, method='average')

    mod4_base = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42)).fit(treino_x_base, treino_p.values)
    rank_4_base = pd.Series(mod4_base.predict(teste_x_base)[0], index=algoritmos).rank(ascending=False)
    
    mod4_ext = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42)).fit(treino_x_ext, treino_p.values)
    rank_4_ext = pd.Series(mod4_ext.predict(teste_x_ext)[0], index=algoritmos).rank(ascending=False)

    mod5_base = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42)).fit(treino_x_base, treino_r.values)
    rank_5_base = pd.Series(mod5_base.predict(teste_x_base)[0], index=algoritmos).rank(ascending=True)
    
    mod5_ext = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42)).fit(treino_x_ext, treino_r.values)
    rank_5_ext = pd.Series(mod5_ext.predict(teste_x_ext)[0], index=algoritmos).rank(ascending=True)

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

print("\n--- ETAPA 4: IMPORTÂNCIA POR ALGORITMO BASE ---")
importancias_dict = {}
for clf_name in algoritmos:
    rf_meta = RandomForestRegressor(random_state=42)
    rf_meta.fit(df_x_ext, df_p[clf_name])
    imps = pd.Series(rf_meta.feature_importances_, index=df_x_ext.columns)
    importancias_dict[clf_name] = imps.sort_values(ascending=False)
    print(f"Top 2 para {clf_name}:\n{importancias_dict[clf_name].head(2)}\n")