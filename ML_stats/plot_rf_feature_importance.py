import pandas as pd
import matplotlib.pyplot as plt

#stats_df = pd.read_csv("feature_importance_WL.csv", sep= ",",index_col=0)
stats_df = pd.read_csv("feature_importance_vig.csv", sep= ",",index_col=0)
print(stats_df.head(1))

stats_df = stats_df[:10]

# reverse the rows
stats_df = stats_df[::-1]

importances = stats_df['score_mean'].tolist()
features = stats_df.index

fig, ax = plt.subplots(figsize=(8,5))
#plt.title('Feature Importances')
ax.barh(features, importances, color='cornflowerblue', align='center')
#plt.yticks(range(max(importances)))
ax.set_xlabel('Relative Importance')
plt.show()