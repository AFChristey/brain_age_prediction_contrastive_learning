import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

method_type = 'yaware' # expw, threshold_reduction_sum, yaware
modality = 'dr' # T1, dr, stiffness
variable_of_interest = 'coverages'  # 'sites', 'sex', 'split', 'imbalances', 'coverages'

#for modality in ['T1', 'dr']:
#    for method_type in ['expw', 'threshold_reduction_sum', 'yaware']:
#        for variable_of_interest in ['sites', 'sex', 'split', 'imbalances', 'coverages']:

df = pd.read_csv('/Users/jakobtraeuble/PycharmProjects/contrastive-brain-age-prediction/output/brain-age-mri/'
                 'embeddings/' + modality + '_resnet18_' + method_type + '_adam_tfnone_lr0.0001_step_step10_rate0.9_temp0.1_wd5e-05_bsz32_views2_trainall_True_kernel_rbf_sigma2.0_f1.0_trial0/'
                 'features.csv')

feature_columns = [col for col in df.columns if col.startswith('feature_')]
features = df[feature_columns].values

print('features.shape:', features.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(features)

print('embedding.shape:', embedding.shape)

embedding_df = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
embedding_df['sex'] = df['sex'].str.upper()
embedding_df['sites'] = df['sites']
embedding_df['split'] = df['split']
embedding_df['imbalances'] = df['imbalances']
embedding_df['coverages'] = df['coverages']
embedding_df['age_labels'] = df['age_labels']

sizes = (embedding_df['age_labels'] - embedding_df['age_labels'].min()) + 10
sizes = (sizes / sizes.max()) * 100

if variable_of_interest in ['sites', 'sex', 'split']:
    if variable_of_interest == 'sites':
        col_pal_str = 'hsv'
        order = 1
    elif variable_of_interest == 'sex':
        col_pal_str = 'coolwarm'
        order = -1
    elif variable_of_interest == 'split':
        col_pal_str = 'Set2'
        order = 1

    color_palette = sns.color_palette(col_pal_str, len(embedding_df[variable_of_interest].unique()))[::order]
    print(embedding_df[variable_of_interest].unique())

    legend = True

    plt.figure(figsize=(8, 8))

elif variable_of_interest == 'imbalances' or variable_of_interest == 'coverages':
    norm = mcolors.Normalize(vmin=embedding_df[variable_of_interest].min(),
                             vmax=embedding_df[variable_of_interest].max())
    if variable_of_interest == 'imbalances':
        colormap = plt.cm.viridis
    elif variable_of_interest == 'coverages':
        colormap = plt.cm.coolwarm

    color_palette = colormap(norm(embedding_df[variable_of_interest].values))

    legend = False

    plt.figure(figsize=(10, 8))

ax = sns.scatterplot(
    x='UMAP 1',
    y='UMAP 2',
    hue=variable_of_interest,
    size=sizes,
    sizes=(20, 200),  # Range for sizes of points
    palette=color_palette,
    data=embedding_df,
    legend=legend,
    alpha=0.5
)

if variable_of_interest == 'imbalances' or variable_of_interest == 'coverages':
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # You have to set a dummy array for the ScalarMappable
    plt.colorbar(sm, ax=ax, label=variable_of_interest)

plt.title(f'{method_type.capitalize()}')
plt.savefig(f'/Users/jakobtraeuble/PycharmProjects/contrastive-brain-age-prediction/output/brain-age-mri/plots/{modality}_{method_type}_umap_{variable_of_interest}.png')
#plt.show()

