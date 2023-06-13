import seaborn as sns
sns.set_theme(style="ticks")
import matplotlib.pyplot as plt
import pandas as pd
p = 2
q = 6
import loadsomresult as ls
file_location = "./somresult/result_"+str(p)+"_"+str(q)+".csv"
# data, labels, data_len = ls.loadsom(file_location,0)

# df = sns.load_dataset("penguins")
df = pd.read_csv(file_location)
df = df.drop("phone_id",axis=1)
print(df)

# copy the data
df_max_scaled = df.copy()

# apply normalization techniques
for column in df_max_scaled.columns:
    if column != "cluster":
        df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max()

# view normalized data
print(df_max_scaled)
sns.color_palette("husl", 6)
# sns.pairplot(df, hue="cluster", palette="pastel")
# plt.savefig("scatter_matrix.png")

# view corelation heatmap
df = df.drop("cluster",axis=1)
a = df.corr()
# print(a)
sns.heatmap(a, 
        xticklabels=a.columns,
        yticklabels=a.columns,
        linewidths=.5,
        annot=True)
plt.savefig("correlation_heatmap.png",bbox_inches='tight')