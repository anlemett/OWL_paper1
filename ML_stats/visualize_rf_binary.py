import pickle
import pydotplus

import graphviz
from sklearn.tree import export_graphviz

#import matplotlib.pyplot as plt
#from sklearn.tree import plot_tree

import pandas as pd

#load the model from disk
filename = 'rf_binary.sav'
rf = pickle.load(open(filename, 'rb'))

############################ Visualize first DT ###############################

#features3 = ['RightBlinkClosingSpeed_mean',
#             'RightBlinkClosingSpeed_max',
#             'LeftBlinkClosingSpeed_std']
#features3 = ['RightPupilDiameter_min',
#            'LeftPupilDiameter_max',
#            'RightBlinkClosingSpeed_median']
    
features3 = ['RightBlinkOpeningAmplitude_mean',
             'HeadRoll_median',
             'HeadPitch_min']

classes = ["low", "high"]
print(rf.classes_) #[1 2]



dot_data = export_graphviz(rf.estimators_[0], 
                           feature_names=features3,
                           class_names=classes, 
                           filled=True, impurity=True, 
                           rounded=True)
#graph = graphviz.Source(dot_data, format='svg')
graph = graphviz.Source(dot_data, format='png')

#fig_filename = 'RF_dt0.svg'
fig_filename = 'RF_dt0'
graph.render(fig_filename)

'''
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,10), dpi=1200)
plot_tree(rf.estimators_[0], 
      feature_names=features3,
      class_names=classes, 
      filled=True, impurity=True, 
      rounded=True)
plt.savefig('RV_dt0.svg', format='svg', dpi=1200, bbox_inches = "tight")
plt.show()
'''
############################ Visualize sample 0 ###############################
tree0 = rf.estimators_[0]

dot_data = export_graphviz(tree0, out_file=None,
                                feature_names=features3,
                                class_names=classes,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.del_node('"\\n"')

# empty all nodes, i.e.set color to white and number of samples to zero
for node in graph.get_node_list():
    if node.get_attributes().get('label') is None:
        continue
    if 'samples = ' in node.get_attributes()['label']:
        labels = node.get_attributes()['label'].split('<br/>')
        for i, label in enumerate(labels):
            if label.startswith('samples = '):
                labels[i] = 'samples = 0'
        node.set('label', '<br/>'.join(labels))
        node.set_fillcolor('white')


filename = "class_low_sample.csv"
sample = pd.read_csv(filename, sep=' ')
decision_paths = tree0.decision_path(sample)


for decision_path in decision_paths:
    for n, node_value in enumerate(decision_path.toarray()[0]):
        if node_value == 0:
            continue
        node = graph.get_node(str(n))[0]            
        node.set_fillcolor('green')
        labels = node.get_attributes()['label'].split('<br/>')
        for i, label in enumerate(labels):
            if label.startswith('samples = '):
                labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

        node.set('label', '<br/>'.join(labels))

#fig_filename = 'RF_dt0_sample0.svg'
#graph.write_svg(fig_filename)

fig_filename = 'RF_dt0_sample0.png'
graph.write_png(fig_filename)

############################ Visualize sample 56 ##############################
tree0 = rf.estimators_[0]

dot_data = export_graphviz(tree0, out_file=None,
                                feature_names=features3,
                                class_names=classes,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.del_node('"\\n"')

# empty all nodes, i.e.set color to white and number of samples to zero
for node in graph.get_node_list():
    if node.get_attributes().get('label') is None:
        continue
    if 'samples = ' in node.get_attributes()['label']:
        labels = node.get_attributes()['label'].split('<br/>')
        for i, label in enumerate(labels):
            if label.startswith('samples = '):
                labels[i] = 'samples = 0'
        node.set('label', '<br/>'.join(labels))
        node.set_fillcolor('white')


filename = "class_high_sample.csv"
sample = pd.read_csv(filename, sep=' ')
decision_paths = tree0.decision_path(sample)


for decision_path in decision_paths:
    for n, node_value in enumerate(decision_path.toarray()[0]):
        if node_value == 0:
            continue
        node = graph.get_node(str(n))[0]            
        node.set_fillcolor('green')
        labels = node.get_attributes()['label'].split('<br/>')
        for i, label in enumerate(labels):
            if label.startswith('samples = '):
                labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

        node.set('label', '<br/>'.join(labels))

#fig_filename = 'RF_dt0_sample56.svg'
#graph.write_svg(fig_filename)

fig_filename = 'RF_dt0_sample56.png'
graph.write_png(fig_filename)



