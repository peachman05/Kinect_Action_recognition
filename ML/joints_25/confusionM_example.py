
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

y_true = ["honda", "chevrolet", "honda", "toyota", "toyota", "chevrolet"]
y_pred = ["honda", "chevrolet", "honda", "toyota", "toyota", "honda"]
data = confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
# fig = plt.figure(figsize = (3,2))
fig, ax = plt.subplots(figsize=(5,5))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
ax.set_ylim(3,0)
# fig.set_size_inches(10, 10)
plt.show()