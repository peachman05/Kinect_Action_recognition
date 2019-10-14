import numpy as np
import pickle


from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from data_helper import reduce_joint_dimension,reform_to_sequence, create_model


sequence_length = 25 # timestep

path_save = "F:/Master Project/Dataset/Extract_Data/25 joints"
f_x = open(path_save+"/test_x.pickle",'rb')
f_y = open(path_save+"/test_y.pickle",'rb')
origin_test_x = pickle.load(f_x)
origin_test_y = np.array(pickle.load(f_y))


origin_test_x = reduce_joint_dimension(origin_test_x,'6')
test_x, test_y  = reform_to_sequence( origin_test_x  , origin_test_y, False, sequence_length)


weights_path = 'weight-sampling-01-0.96.hdf5' # 15 frame
max_frame = 25
num_joint = 6
model = create_model(max_frame, num_joint*3, 4)
model.load_weights(weights_path)
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#### Accuracy
loss, acc = model.evaluate(test_x, y = test_y, batch_size=384, verbose=0)
print(loss,acc)


#### Confusion Matrix
y_pred=model.predict_classes(test_x)
normalize = True
cm = confusion_matrix(test_y, y_pred)
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
else:
    print('Confusion matrix, without normalization')

# classes = ['拍球','投球','传球','站立']
classes = ['dribble','shoot','pass','stand']
# title = 'test'
# cmap=plt.cm.Blues
# fig, ax = plt.subplots()
# im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
# ax.figure.colorbar(im, ax=ax)
# # We want to show all ticks...
# # ax.figure.set_size_inches(10, 10)
# ax.set(xticks=np.arange(cm.shape[1]),
#        yticks=np.arange(cm.shape[0]),
#        # ... and label them with the respective list entries
#        xticklabels=classes, yticklabels=classes,
#        title=title,
#        ylabel='True label',
#        xlabel='Predicted label')

# # Loop over data dimensions and create text annotations.
# fmt = '.2f' if normalize else 'd'
# thresh = cm.max() / 2.
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(j, i, format(cm[i, j], fmt),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black")
# fig.tight_layout()
# plt.show()

df_cm = pd.DataFrame(cm, columns=classes, index=classes)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=(5,5))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,fmt=".2f", annot_kws={"size": 16})# font size
ax.set_ylim(5, 0)
plt.show()

