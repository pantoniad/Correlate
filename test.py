import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Example dataset
np.random.seed(0)
df = pd.DataFrame({
    'Value': np.concatenate([
        np.random.normal(10, 2, 15),
        np.random.normal(15, 2, 15),
        np.random.normal(20, 2, 15)
    ]),
    'Group': ['A']*15 + ['B']*15 + ['C']*15
})

# 1. ANOVA
model = ols('Value ~ C(Group)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 2. Tukey HSD for mean separation
tukey = pairwise_tukeyhsd(endog=df['Value'], groups=df['Group'], alpha=0.05)
print(tukey)

# 3. Plot mean separation
means = df.groupby('Group')['Value'].mean()
errors = df.groupby('Group')['Value'].sem()  # standard error of mean

fig, ax = plt.subplots()
means.plot(kind='bar', yerr=errors, capsize=5, ax=ax, color='skyblue')

# Annotate with Tukey results
# Get group letters (simple manual mapping here, can be automated)
letters = {'A': 'a', 'B': 'b', 'C': 'c'}
for i, group in enumerate(means.index):
    ax.text(i, means[group] + errors[group] + 0.5, letters[group],
            ha='center', fontsize=12)

ax.set_ylabel('Mean Value')
ax.set_title('Mean Separation Plot (Tukey HSD)')
plt.show()
