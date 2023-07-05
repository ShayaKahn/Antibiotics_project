import os
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\Antibiotics_project')
import pandas as pd
import numpy as np
from subject_class import Subject
from dash import Dash, html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from optimal_cohort import OptimalCohort
from overlap import Overlap
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
from Jaccard_disappeared_species import JaccardDisappearedSpecies
from compare_to_shuffled import ShuffledVsNormal

# Functions
def normalize_cohort(cohort):
    cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
    return cohort_normalized

def filter_data(df):
    # filter the data
    columns_to_check = df.columns[1:]
    mean_values = df[columns_to_check].mean(axis=1)
    condition_mask = mean_values >= 0.0001
    df = df[condition_mask]
    return df

def create_barplots(baseline_cohort, end_cohort, overlap_type, names, rows, cols):
    results = []
    for sample_future in end_cohort:
        sample_results = []
        for sample_base in baseline_cohort:
            J_object = Overlap(sample_base, sample_future, overlap_type=overlap_type)
            sample_results.append(J_object.calculate_overlap())
        results.append(sample_results)

    x = np.arange(1, len(names)+1, 1)

    similarity_barplots = make_subplots(rows=rows, cols=cols, subplot_titles=names)

    # Iterate over results and add bar plots to subplots
    for i, result in enumerate(results):
        row = (i // cols) + 1
        col = (i % cols) + 1
        similarity_barplots.add_trace(
            go.Bar(x=np.hstack((x[0:i], x[i + 1:])), y=np.hstack((result[0:i], result[i + 1:])),
                   marker_color='blue', name='All Results'), row=row, col=col)
        similarity_barplots.add_trace(
            go.Bar(x=[x[i]], y=[result[i]], marker_color='red', name='Selected Result'),
            row=row, col=col)

        # Update x-axis properties for all subplots
    similarity_barplots.update_xaxes(linecolor='black', linewidth=2, mirror=False)

    # Update y-axis properties for all subplots
    similarity_barplots.update_yaxes(linecolor='black', linewidth=2, mirror=False)

    # Update layout
    similarity_barplots.update_layout(height=1000, width=1200, showlegend=False, plot_bgcolor='white')
    return similarity_barplots, results

def confidence_ellipse(x, y, n_std=1.96, size=100, **kwargs):
    """
    Get a Plotly scatter trace representing the covariance confidence ellipse of `x` and `y`.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    size : int
        Number of points defining the ellipse
    **kwargs
        Additional keyword arguments to be passed to `plotly.graph_objects.Scatter`

    Returns
    -------
    plotly.graph_objects.Scatter
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack([ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)])

    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4)],
                                [-np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    scale_matrix = np.array([[x_scale, 0],
                             [0, y_scale]])
    ellipse_coords = ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix

    x_ellipse = ellipse_coords[:, 0]
    y_ellipse = ellipse_coords[:, 1]

    return go.Scatter(x=x_ellipse, y=y_ellipse, mode='lines', **kwargs)

### Post-Antibiotic Gut Mucosal Microbiome Reconstitution Is Impaired by Probiotics and Improved by Autologous FMT ###

df = pd.read_excel('Metaphlan_stool.xlsx')
Species_column = df[df.columns[0]]
Species = np.array(list(Species_column))

# aFMT data

aFMT_baseline_1 = df.iloc[:, 1:8].values  # 1,2,3,4,5,6,7 of baseline.
aFMT_antibiotics_1 = df.iloc[:, 8].values   # day 2 of antibiotics.
aFMT_intervention_1 = df.iloc[:, 9:19].values  # days 1,2,3,4,5,14,21,28,42,56 of intervention.
aFMT_month_1 = df.iloc[:, 19:24].values  # months after the end of intervention 1,2,3,4,6.

aFMT_baseline_2 = df.iloc[:, 24:30].values  # 1,2,3,5,6,7 of baseline.
aFMT_antibiotics_2 = df.iloc[:, 30:35].values   # days 2,4,5,6,7  of antibiotics.
aFMT_intervention_2 = df.iloc[:, 35:47].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
aFMT_month_2 = df.iloc[:, 47:52].values  # months after the end of intervention 2,3,4,5,6.

aFMT_baseline_3 = df.iloc[:, 52:58].values  # 1,2,3,4,5,7 of baseline.
aFMT_antibiotics_3 = df.iloc[:, 58:64].values   # days 2,3,4,5,6,7  of antibiotics.
aFMT_intervention_3 = df.iloc[:, 64:76].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
aFMT_month_3 = df.iloc[:, 76:80].values  # months after the end of intervention 2,3,4,5.

aFMT_baseline_4 = df.iloc[:, 80:86].values  # 1,3,4,5,6,7 of baseline.
aFMT_antibiotics_4 = df.iloc[:, 86:91].values   # days 1,2,5,6,7  of antibiotics.
aFMT_intervention_4 = df.iloc[:, 91:102].values  # days 3,4,5,6,7,14,21,24,28,42,56 of intervention.
aFMT_intervention_4 = np.delete(aFMT_intervention_4, 7, axis=1)
aFMT_month_4 = df.iloc[:, 102:106].values  # months after the end of intervention 3,4,5,6.

aFMT_baseline_5 = df.iloc[:, 106:113].values  # 1,2,3,4,5,6,7 of baseline.
aFMT_antibiotics_5 = df.iloc[:, 113:119].values   # days 1,2,3,4,5,6  of antibiotics.
aFMT_intervention_5 = df.iloc[:, 119:131].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
aFMT_month_5 = df.iloc[:, 131:134].values  # months after the end of intervention 4,5,6.

aFMT_baseline_6 = df.iloc[:, 134:141].values  # 1,2,3,4,5,6,7 of baseline.
aFMT_antibiotics_6 = df.iloc[:, 141:147].values   # days 2,3,4,5,6,7  of antibiotics.
aFMT_intervention_6 = df.iloc[:, 147:159].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.

# probiotics data

pro_baseline_1 = df.iloc[:, 159:166].values  # 1,2,3,4,5,6,7 of baseline.
pro_antibiotics_1 = df.iloc[:, 166:173].values   # days 1,2,3,4,5,6,7  of antibiotics.
pro_intervention_1 = df.iloc[:, 173:185].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
pro_month_1 = df.iloc[:, 185:190].values  # months after the end of intervention 2,3,4,5,6

pro_baseline_2 = df.iloc[:, 190:195].values  # 1,2,3,4,5 of baseline.
pro_antibiotics_2 = df.iloc[:, 195:201].values   # days 2,3,4,5,6,7  of antibiotics.
pro_intervention_2 = df.iloc[:, 201:208].values  # days 1,2,4,5,6,14,21 of intervention.
pro_month_2 = df.iloc[:, 208:212].values  # months after the end of intervention 2,3,4,5.

pro_baseline_3 = df.iloc[:, 212:218].values  # 1,2,3,4,5,7 of baseline.
pro_antibiotics_3 = df.iloc[:, 218:223].values   # days 1,2,3,4,6  of antibiotics.
pro_intervention_3 = df.iloc[:, 223:234].values  # days 1,2,3,4,5,6,14,21,28,42,56 of intervention.
pro_month_3 = df.iloc[:, 234:237].values  # months after the end of intervention 3,5,6.

pro_baseline_4 = df.iloc[:, 237:244].values  # 1,2,3,4,5,6,7 of baseline.
pro_antibiotics_4 = df.iloc[:, 244:251].values   # days 1,2,3,4,5,6,7  of antibiotics.
pro_intervention_4 = df.iloc[:, 251:261].values  # days 1,2,4,5,6,14,21,28,42,56 of intervention.
pro_month_4 = df.iloc[:, 261:265].values  # months after the end of intervention 3,4,5,6.

pro_baseline_5 = df.iloc[:, 265:272].values  # 1,2,3,4,5,6,7 of baseline.
pro_antibiotics_5 = df.iloc[:, 272:274].values   # days 1,2 of antibiotics.
pro_intervention_5 = df.iloc[:, 274:286].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
pro_month_5 = df.iloc[:, 286:289].values  # months after the end of intervention 4,5,6.

pro_baseline_6 = df.iloc[:, 289:296].values  # 1,2,3,4,5,6,7 of baseline.
pro_antibiotics_6 = df.iloc[:, 296:302].values   # days 1,2,4,5,6,7 of antibiotics.
pro_intervention_6 = df.iloc[:, 302:314].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.

pro_baseline_7 = df.iloc[:, 314:321].values  # 1,2,3,4,5,6,7 of baseline.
pro_antibiotics_7 = df.iloc[:, 321:328].values   # days 1,2,3,4,5,6,7 of antibiotics.
pro_intervention_7 = df.iloc[:, 328:340].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.

pro_baseline_8 = df.iloc[:, 340:343].values  # 1,2,3 of baseline.
pro_antibiotics_8 = df.iloc[:, 343:347].values   # days 2,4,6,7 of antibiotics.
pro_intervention_8 = df.iloc[:, 347:356].values  # days 3,5,6,7,14,21,28,42,56 of intervention.

# spontaneous data

spo_baseline_1 = df.iloc[:, 356:361].values  # 1,2,3,4,7 of baseline.
spo_antibiotics_1 = df.iloc[:, 361:367].values   # days 2,3,4,5,6,7 of antibiotics.
spo_intervention_1 = df.iloc[:, 367:376].values  # days 3,4,5,7,14,21,28,42,56 of intervention.
spo_month_1 = df.iloc[:, 376:380].values  # months after the end of intervention 3,4,5,6.

spo_baseline_2 = df.iloc[:, 380:387].values  # 1,2,3,4,5,6,7 of baseline.
spo_antibiotics_2 = df.iloc[:, 387:394].values   # days 1,2,3,4,5,6,7 of antibiotics.
spo_intervention_2 = df.iloc[:, 394:406].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
spo_month_2 = df.iloc[:, 406:410].values  # months after the end of intervention 3,4,5,6.

spo_baseline_3 = df.iloc[:, 410:417].values  # 1,2,3,4,5,6,7 of baseline.
spo_antibiotics_3 = df.iloc[:, 417:424].values   # days 1,2,3,4,5,6,7 of antibiotics.
spo_intervention_3 = df.iloc[:, 424:436].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
spo_month_3 = df.iloc[:, 436:439].values  # months after the end of intervention 4,5,6.

spo_baseline_4 = df.iloc[:, 439:445].values  # 1,2,3,4,6,7 of baseline.
spo_antibiotics_4 = df.iloc[:, 445:452].values   # days 1,2,3,4,5,6,7 of antibiotics.
spo_intervention_4 = df.iloc[:, 452:464].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.

spo_baseline_5 = df.iloc[:, 464:471].values  # 1,2,3,4,5,6,7 of baseline.
spo_antibiotics_5 = df.iloc[:, 471:478].values   # days 1,2,3,4,5,6,7 of antibiotics.
spo_intervention_5 = df.iloc[:, 478:490].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
spo_month_5 = df.iloc[:, 490:493].values  # months after the end of intervention 4,5,6.

spo_baseline_6 = df.iloc[:, 493:498].values  # 2,3,4,6,7 of baseline.
spo_antibiotics_6 = df.iloc[:, 498:504].values   # days 2,3,4,5,6,7 of antibiotics.
spo_intervention_6 = df.iloc[:, 504:515].values  # days 1,2,3,4,6,7,14,21,28,42,56 of intervention.

spo_baseline_7 = df.iloc[:, 515:522].values  # 1,2,3,4,5,6,7 of baseline.
spo_antibiotics_7 = df.iloc[:, 522:529].values   # days 1,2,3,4,5,6,7 of antibiotics.
spo_intervention_7 = df.iloc[:, 529:541].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.

aFMT_subject_1 = Subject(base_array=aFMT_baseline_1, ant_array=aFMT_antibiotics_1, int_array=aFMT_intervention_1,
                         month_array=aFMT_month_1, base_days=['1', '2', '3', '4', '5', '6', '7'], ant_days=['2'],
                         int_days=['1', '2', '3' ,'4' ,'5' ,'14' ,'21' ,'28' ,'42' ,'56'],
                         month_number=['1', '2', '3', '4', '6'])
aFMT_subject_1_base_dict = aFMT_subject_1.create_base_dict()
aFMT_subject_1_ant_dict = aFMT_subject_1.create_ant_dict()
aFMT_subject_1_int_dict = aFMT_subject_1.create_int_dict()
aFMT_subject_1_month_dict = aFMT_subject_1.create_month_dict()

aFMT_subject_2 = Subject(base_array=aFMT_baseline_2, ant_array=aFMT_antibiotics_2, int_array=aFMT_intervention_2,
                         month_array=aFMT_month_2, base_days=['1', '2', '3', '5', '6', '7'],
                         ant_days=['2', '4', '5', '6', '7'],
                         int_days=['1', '2', '3' ,'4' ,'5', '6', '7' ,'14' ,'21' ,'28' ,'42' ,'56'],
                         month_number=['2', '3', '4', '5', '6'])
aFMT_subject_2_base_dict = aFMT_subject_2.create_base_dict()
aFMT_subject_2_ant_dict = aFMT_subject_2.create_ant_dict()
aFMT_subject_2_int_dict = aFMT_subject_2.create_int_dict()
aFMT_subject_2_month_dict = aFMT_subject_2.create_month_dict()

aFMT_subject_3 = Subject(base_array=aFMT_baseline_3, ant_array=aFMT_antibiotics_3, int_array=aFMT_intervention_3,
                         month_array=aFMT_month_3, base_days=['1', '2', '3', '4', '5', '7'],
                         ant_days=['2', '3', '4', '5', '6', '7'],
                         int_days=['1', '2', '3' ,'4' ,'5', '6', '7' , '14' ,'21' ,'28' ,'42' ,'56'],
                         month_number=['2', '3', '4', '5'])
aFMT_subject_3_base_dict = aFMT_subject_3.create_base_dict()
aFMT_subject_3_ant_dict = aFMT_subject_3.create_ant_dict()
aFMT_subject_3_int_dict = aFMT_subject_3.create_int_dict()
aFMT_subject_3_month_dict = aFMT_subject_3.create_month_dict()

aFMT_subject_4 = Subject(base_array=aFMT_baseline_4, ant_array=aFMT_antibiotics_4, int_array=aFMT_intervention_4,
                         month_array=aFMT_month_4, base_days=['1', '3', '4', '5' ,'6' , '7'],
                         ant_days=['1', '2', '5', '6', '7'],
                         int_days=['3' ,'4' ,'5', '6', '7' ,'14' ,'21' ,'28' ,'42' ,'56'],
                         month_number=['3', '4', '5', '6'])
aFMT_subject_4_base_dict = aFMT_subject_4.create_base_dict()
aFMT_subject_4_ant_dict = aFMT_subject_4.create_ant_dict()
aFMT_subject_4_int_dict = aFMT_subject_4.create_int_dict()
aFMT_subject_4_month_dict = aFMT_subject_4.create_month_dict()

aFMT_subject_5 = Subject(base_array=aFMT_baseline_5, ant_array=aFMT_antibiotics_5, int_array=aFMT_intervention_5,
                         month_array=aFMT_month_5, base_days=['1', '2', '3', '4', '5', '6', '7'],
                         ant_days=['1', '2', '3', '4', '5', '6'],
                         int_days=['1', '2', '3' ,'4' ,'5', '6', '7' , '14' ,'21' ,'28' ,'42' ,'56'],
                         month_number=['4', '5', '6'])
aFMT_subject_5_base_dict = aFMT_subject_5.create_base_dict()
aFMT_subject_5_ant_dict = aFMT_subject_5.create_ant_dict()
aFMT_subject_5_int_dict = aFMT_subject_5.create_int_dict()
aFMT_subject_5_month_dict = aFMT_subject_5.create_month_dict()

aFMT_subject_6 = Subject(base_array=aFMT_baseline_6, ant_array=aFMT_antibiotics_6, int_array=aFMT_intervention_6,
                         base_days=['1', '2', '3', '4', '5', '6', '7'], ant_days=['2', '3', '4', '5', '6', '7'],
                         int_days=['1', '2', '3' ,'4' ,'5', '6', '7', '14' ,'21' ,'28' ,'42' ,'56'])
aFMT_subject_6_base_dict = aFMT_subject_6.create_base_dict()
aFMT_subject_6_ant_dict = aFMT_subject_6.create_ant_dict()
aFMT_subject_6_int_dict = aFMT_subject_6.create_int_dict()

pro_subject_1 = Subject(base_array=pro_baseline_1, ant_array=pro_antibiotics_1 ,int_array=pro_intervention_1,
                        month_array=pro_month_1,
                        base_days=['1', '2', '3', '4', '5', '6', '7'], ant_days=['1', '2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '3' ,'4' ,'5', '6', '7' ,'14' ,'21' ,'28' ,'42' ,'56'],
                        month_number=['2', '3', '4', '5', '6'])
pro_subject_1_base_dict = pro_subject_1.create_base_dict()
pro_subject_1_ant_dict = pro_subject_1.create_ant_dict()
pro_subject_1_int_dict = pro_subject_1.create_int_dict()
pro_subject_1_month_dict = pro_subject_1.create_month_dict()

pro_subject_2 = Subject(base_array=pro_baseline_2, ant_array=pro_antibiotics_2, int_array=pro_intervention_2,
                        month_array=pro_month_2,
                        base_days=['1', '2', '3', '4', '5'], ant_days=['2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '4', '5', '6', '14', '21'], month_number=['2', '3', '4', '5'])
pro_subject_2_base_dict = pro_subject_2.create_base_dict()
pro_subject_2_ant_dict = pro_subject_2.create_ant_dict()
pro_subject_2_int_dict = pro_subject_2.create_int_dict()
pro_subject_2_month_dict = pro_subject_2.create_month_dict()

pro_subject_3 = Subject(base_array=pro_baseline_3, ant_array=pro_antibiotics_3, int_array=pro_intervention_3,
                        month_array=pro_month_3,
                        base_days=['1', '2', '3', '4', '5', '7'], ant_days=['1', '2', '3', '4', '6'],
                        int_days=['1', '2', '3', '4', '5', '6', '14', '21', '28', '42', '56'], month_number=['3', '5', '6'])
pro_subject_3_base_dict = pro_subject_3.create_base_dict()
pro_subject_3_ant_dict = pro_subject_3.create_ant_dict()
pro_subject_3_int_dict = pro_subject_3.create_int_dict()
pro_subject_3_month_dict = pro_subject_3.create_month_dict()

pro_subject_4 = Subject(base_array=pro_baseline_4, ant_array=pro_antibiotics_4, int_array=pro_intervention_4,
                        month_array=pro_month_4,
                        base_days=['1', '2', '3', '4', '5', '6', '7'], ant_days=['1', '2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '4', '5', '6', '14', '21', '28', '42', '56'], month_number=['3', '4', '5', '6'])
pro_subject_4_base_dict = pro_subject_4.create_base_dict()
pro_subject_4_ant_dict = pro_subject_4.create_ant_dict()
pro_subject_4_int_dict = pro_subject_4.create_int_dict()
pro_subject_4_month_dict = pro_subject_4.create_month_dict()

pro_subject_5 = Subject(base_array=pro_baseline_5, ant_array=pro_antibiotics_5, int_array=pro_intervention_5,
                        month_array=pro_month_5,
                        base_days=['1', '2', '3', '4', '5', '6', '7'], ant_days=['1', '2'],
                        int_days=['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'],
                        month_number=['4', '5', '6'])
pro_subject_5_base_dict = pro_subject_5.create_base_dict()
pro_subject_5_ant_dict = pro_subject_5.create_ant_dict()
pro_subject_5_int_dict = pro_subject_5.create_int_dict()
pro_subject_5_month_dict = pro_subject_5.create_month_dict()

pro_subject_6 = Subject(base_array=pro_baseline_6, ant_array=pro_antibiotics_6, int_array=pro_intervention_6,
                        base_days=['1', '2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'],
                        ant_days=['1', '2', '4', '5', '6', '7'])
pro_subject_6_int_dict = pro_subject_6.create_int_dict()
pro_subject_6_base_dict = pro_subject_6.create_base_dict()
pro_subject_6_ant_dict = pro_subject_6.create_ant_dict()

pro_subject_7 = Subject(base_array=pro_baseline_7, ant_array=pro_antibiotics_7, int_array=pro_intervention_7,
                        base_days=['1', '2', '3', '4', '5', '6', '7'], ant_days=['1', '2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'])
pro_subject_7_base_dict = pro_subject_7.create_base_dict()
pro_subject_7_ant_dict = pro_subject_7.create_ant_dict()
pro_subject_7_int_dict = pro_subject_7.create_int_dict()

pro_subject_8 = Subject(base_array=pro_baseline_8, int_array=pro_intervention_8, ant_array=pro_antibiotics_8,
                        base_days=['1', '2', '3'] ,int_days=['3', '5', '6', '7', '14', '21', '28', '42', '56'], ant_days=
                        ['2', '4', '6', '7'])
pro_subject_8_int_dict = pro_subject_8.create_int_dict()
pro_subject_8_base_dict = pro_subject_8.create_base_dict()
pro_subject_8_ant_dict = pro_subject_8.create_ant_dict()

spo_subject_1 = Subject(base_array=spo_baseline_1, ant_array=spo_antibiotics_1, int_array=spo_intervention_1,
                        month_array=spo_month_1, base_days=['1', '2', '3', '4', '7'],
                        ant_days=['2', '3', '4', '5', '6', '7'],
                        int_days=['3', '4', '5', '7', '14', '21', '28', '42', '56'], month_number=['3', '4', '5', '6'])
spo_subject_1_base_dict = spo_subject_1.create_base_dict()
spo_subject_1_ant_dict = spo_subject_1.create_ant_dict()
spo_subject_1_int_dict = spo_subject_1.create_int_dict()
spo_subject_1_month_dict = spo_subject_1.create_month_dict()

spo_subject_2 = Subject(base_array=spo_baseline_2, ant_array=spo_antibiotics_2, int_array=spo_intervention_2,
                        month_array=spo_month_2, base_days=['1', '2', '3', '4', '5', '6', '7'],
                        ant_days=['1', '2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'],
                        month_number=['3', '4', '5', '6'])
spo_subject_2_base_dict = spo_subject_2.create_base_dict()
spo_subject_2_ant_dict = spo_subject_2.create_ant_dict()
spo_subject_2_int_dict = spo_subject_2.create_int_dict()
spo_subject_2_month_dict = spo_subject_2.create_month_dict()

spo_subject_3 = Subject(base_array=spo_baseline_3, ant_array=spo_antibiotics_3, int_array=spo_intervention_3,
                        month_array=spo_month_3, base_days=['1', '2', '3', '4', '5', '6', '7'],
                        ant_days=['1', '2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'],
                        month_number=['4', '5', '6'])
spo_subject_3_base_dict = spo_subject_3.create_base_dict()
spo_subject_3_ant_dict = spo_subject_3.create_ant_dict()
spo_subject_3_int_dict = spo_subject_3.create_int_dict()
spo_subject_3_month_dict = spo_subject_3.create_month_dict()

spo_subject_4 = Subject(base_array=spo_baseline_4, ant_array=spo_antibiotics_4, int_array=spo_intervention_4,
                        base_days=['1', '2', '3', '4', '6', '7'], ant_days=['1', '2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'])
spo_subject_4_base_dict = spo_subject_4.create_base_dict()
spo_subject_4_ant_dict = spo_subject_4.create_ant_dict()
spo_subject_4_int_dict = spo_subject_4.create_int_dict()

spo_subject_5 = Subject(base_array=spo_baseline_5, ant_array=spo_antibiotics_5, int_array=spo_intervention_5,
                        month_array=spo_month_5, base_days=['1', '2', '3', '4', '5', '6', '7'],
                        ant_days=['1', '2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'],
                        month_number=['4', '5', '6'])
spo_subject_5_base_dict = spo_subject_5.create_base_dict()
spo_subject_5_ant_dict = spo_subject_5.create_ant_dict()
spo_subject_5_int_dict = spo_subject_5.create_int_dict()
spo_subject_5_month_dict = spo_subject_5.create_month_dict()

spo_subject_6 = Subject(base_array=spo_baseline_6, ant_array=spo_antibiotics_6, int_array=spo_intervention_6,
                        base_days=['2', '3', '4', '6', '7'],
                        ant_days=['2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '3', '4', '6', '7', '14', '21', '28', '42', '56'])
spo_subject_6_base_dict = spo_subject_6.create_base_dict()
spo_subject_6_ant_dict = spo_subject_6.create_ant_dict()
spo_subject_6_int_dict = spo_subject_6.create_int_dict()

spo_subject_7 = Subject(base_array=spo_baseline_7, ant_array=spo_antibiotics_7, int_array=spo_intervention_7,
                        base_days=['1', '2', '3', '4', '5', '6', '7'],
                        ant_days=['1', '2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'])
spo_subject_7_base_dict = spo_subject_7.create_base_dict()
spo_subject_7_ant_dict = spo_subject_7.create_ant_dict()
spo_subject_7_int_dict = spo_subject_7.create_int_dict()

# Create cohorts

baseline_dict = {'aFMT_1': aFMT_baseline_1.T, 'aFMT_2': aFMT_baseline_2.T, 'aFMT_3': aFMT_baseline_3.T,
                 'aFMT_4': aFMT_baseline_4.T, 'aFMT_5': aFMT_baseline_5.T, 'aFMT_6': aFMT_baseline_6[:, 2:].T,
                 'pro_1': pro_baseline_1.T, 'pro_2': pro_baseline_2.T, 'pro_3': pro_baseline_3.T,
                 'pro_4': pro_baseline_4.T, 'pro_5': pro_baseline_5.T, 'pro_6': pro_baseline_6.T,
                 'pro_7': pro_baseline_7.T, 'pro_8': pro_baseline_8.T, 'spo_1': spo_baseline_1.T,
                 'spo_2': spo_baseline_2.T, 'spo_3': spo_baseline_3.T, 'spo_4': spo_baseline_4.T,
                 'spo_5': spo_baseline_5.T, 'spo_6': spo_baseline_6.T, 'spo_7': spo_baseline_7.T}

baseline_list = list(baseline_dict.values())
baseline_total_matrix = np.concatenate(baseline_list, axis=0)
baseline_total_matrix = normalize_cohort(baseline_total_matrix)

baseline_cohort_opt = OptimalCohort(baseline_dict)
baseline_cohort, chosen_indices = baseline_cohort_opt.get_optimal_samples()

Future_cohort = np.vstack([aFMT_subject_1_month_dict['6'], aFMT_subject_2_month_dict['6'], aFMT_subject_3_month_dict['5'],
                           aFMT_subject_4_month_dict['6'], aFMT_subject_5_month_dict['6'], aFMT_subject_6_int_dict['56'],
                           pro_subject_1_month_dict['6'], pro_subject_2_month_dict['5'], pro_subject_3_month_dict['6'],
                           pro_subject_4_month_dict['6'], pro_subject_5_month_dict['6'], pro_subject_6_int_dict['56'],
                           pro_subject_7_int_dict['56'], pro_subject_8_int_dict['56'], spo_subject_1_month_dict['6'],
                           spo_subject_2_month_dict['6'], spo_subject_3_month_dict['6'], spo_subject_4_int_dict['56'],
                           spo_subject_5_month_dict['6'], spo_subject_6_int_dict['56'], spo_subject_7_int_dict['56']])

Future_cohort_spo = np.vstack([spo_subject_1_month_dict['6'], spo_subject_2_month_dict['6'],
                               spo_subject_3_month_dict['6'], spo_subject_4_int_dict['56'],
                               spo_subject_5_month_dict['6'], spo_subject_6_int_dict['56'],
                               spo_subject_7_int_dict['56']])

Future_cohort = normalize_cohort(Future_cohort)

ABX_cohort = np.vstack([aFMT_subject_1_ant_dict['2'], aFMT_subject_2_ant_dict['7'], aFMT_subject_3_ant_dict['7'],
                        aFMT_subject_4_ant_dict['7'], aFMT_subject_5_ant_dict['6'], aFMT_subject_6_ant_dict['7'],
                        pro_subject_1_ant_dict['7'], pro_subject_2_ant_dict['7'], pro_subject_3_ant_dict['6'],
                        pro_subject_4_ant_dict['7'], pro_subject_5_ant_dict['2'], pro_subject_6_ant_dict['7'],
                        pro_subject_7_ant_dict['7'], pro_subject_8_ant_dict['7'], spo_subject_1_ant_dict['7'],
                        spo_subject_2_ant_dict['7'], spo_subject_3_ant_dict['7'], spo_subject_4_ant_dict['7'],
                        spo_subject_5_ant_dict['7'], spo_subject_6_ant_dict['7'], spo_subject_7_ant_dict['7']])

ABX_dict = {'aFMT_1': aFMT_antibiotics_1.reshape(-1, 1).T, 'aFMT_2': aFMT_antibiotics_2.T, 'aFMT_3': aFMT_antibiotics_3.T,
            'aFMT_4': aFMT_antibiotics_4.T, 'aFMT_5': aFMT_antibiotics_5.T, 'aFMT_6': aFMT_antibiotics_6.T,
            'pro_1': pro_antibiotics_1.T, 'pro_2': pro_antibiotics_2.T, 'pro_3': pro_antibiotics_3.T,
            'pro_4': pro_antibiotics_4.T, 'pro_5': pro_antibiotics_5.T, 'pro_6': pro_antibiotics_6.T,
            'pro_7': pro_antibiotics_7.T, 'pro_8': pro_antibiotics_8.T, 'spo_1': spo_antibiotics_1.T,
            'spo_2': spo_antibiotics_2.T, 'spo_3': spo_antibiotics_3.T, 'spo_4': spo_antibiotics_4.T,
            'spo_5': spo_antibiotics_5.T, 'spo_6': spo_antibiotics_6.T, 'spo_7': spo_antibiotics_7.T}

ABX_list = list(ABX_dict.values())
ABX_total_matrix = np.concatenate(ABX_list, axis=0)
ABX_total_matrix = normalize_cohort(ABX_total_matrix)

names = ['aFMT Subject 1', 'aFMT Subject 2', 'aFMT Subject 3', 'aFMT Subject 4', 'aFMT Subject 5', 'aFMT Subject 6',
         'Pro Subject 1', 'Pro Subject 2', 'Pro Subject 3', 'Pro Subject 4', 'Pro Subject 5', 'Pro Subject 6',
         'Pro Subject 7', 'Pro Subject 8', 'Spo Subject 1', 'Spo Subject 2', 'Spo Subject 3', 'Spo Subject 4',
         'Spo Subject 5', 'Spo Subject 6', 'Spo Subject 7']

# Strict

Jaccard_sets_container = []
for base, future, abx in zip(baseline_cohort, Future_cohort, ABX_cohort):
    object = JaccardDisappearedSpecies(base, future, abx, strict=True)
    Jaccard_set = []
    for sample_base in baseline_cohort:
        Jaccard_set.append(object.calc_jaccard(sample_base))
    Jaccard_sets_container.append(Jaccard_set)

x = np.arange(1, len(Jaccard_sets_container)+1, 1)

similarity_barplots_dis_first_data = make_subplots(rows=7, cols=3, subplot_titles=names)

for i, result in enumerate(Jaccard_sets_container):
    row = (i // 3) + 1
    col = (i % 3) + 1
    similarity_barplots_dis_first_data.add_trace(
        go.Bar(x=np.hstack((x[0:i], x[i + 1:])), y=np.hstack((result[0:i], result[i + 1:])),
                marker_color='blue', name='All Results'), row=row, col=col)
    similarity_barplots_dis_first_data.add_trace(
        go.Bar(x=[x[i]], y=[result[i]], marker_color='red', name='Selected Result'),
        row=row, col=col)

# Update layout
# Update x-axis properties for all subplots
similarity_barplots_dis_first_data.update_xaxes(linecolor='black', linewidth=2, mirror=False)

# Update y-axis properties for all subplots
similarity_barplots_dis_first_data.update_yaxes(linecolor='black', linewidth=2, mirror=False)

# Update layout
similarity_barplots_dis_first_data.update_layout(height=1000, width=1200, showlegend=False, plot_bgcolor='white')

# No strict

Jaccard_sets_container_ns = []
for base, future, abx in zip(baseline_cohort, Future_cohort, ABX_cohort):
    object_ns = JaccardDisappearedSpecies(base, future, abx, strict=False)
    Jaccard_set_ns = []
    for sample_base in baseline_cohort:
        Jaccard_set_ns.append(object_ns.calc_jaccard(sample_base))
    Jaccard_sets_container_ns.append(Jaccard_set_ns)

x = np.arange(1, len(Jaccard_sets_container_ns)+1, 1)

similarity_barplots_dis_first_data_ns = make_subplots(rows=7, cols=3, subplot_titles=names)

for i, result in enumerate(Jaccard_sets_container_ns):
    row = (i // 3) + 1
    col = (i % 3) + 1
    similarity_barplots_dis_first_data_ns.add_trace(
        go.Bar(x=np.hstack((x[0:i], x[i + 1:])), y=np.hstack((result[0:i], result[i + 1:])),
                marker_color='blue', name='All Results'), row=row, col=col)
    similarity_barplots_dis_first_data_ns.add_trace(
        go.Bar(x=[x[i]], y=[result[i]], marker_color='red', name='Selected Result'),
        row=row, col=col)

# Update layout
# Update x-axis properties for all subplots
similarity_barplots_dis_first_data_ns.update_xaxes(linecolor='black', linewidth=2, mirror=False)

# Update y-axis properties for all subplots
similarity_barplots_dis_first_data_ns.update_yaxes(linecolor='black', linewidth=2, mirror=False)

# Update layout
similarity_barplots_dis_first_data_ns.update_layout(height=1000, width=1200, showlegend=False, plot_bgcolor='white')

nonzero_list_ABX = []
nonzero_list_ABX_counts = []
for sample in ABX_cohort:
    nonzero_list_ABX.append(np.nonzero(sample))
    nonzero_list_ABX_counts.append(np.size(np.nonzero(sample)))
nonzero_list_base = []
nonzero_list_base_counts = []
for sample in baseline_cohort:
    nonzero_list_base.append(np.nonzero(sample))
    nonzero_list_base_counts.append(np.size(np.nonzero(sample)))
nonzero_list_future = []
nonzero_list_future_counts = []
for sample in Future_cohort:
    nonzero_list_future.append(np.nonzero(sample))
    nonzero_list_future_counts.append(np.size(np.nonzero(sample)))

intersect = []
intersect_indexes = []

for base, future in zip(nonzero_list_base, nonzero_list_future):
    intersect.append(len(np.intersect1d(base[0], future[0])))
    intersect_indexes.append(np.intersect1d(base[0], future[0]))

intersect_frac_base = []
intersect_frac_future = []
for sample_base, sample_future, inter in zip(baseline_cohort, Future_cohort, intersect_indexes):
    intersect_frac_base.append(np.sum(sample_base[inter]))
    intersect_frac_future.append(np.sum(sample_future[inter]))

intersect_abx_base = [np.intersect1d(base, abx) for base, abx in zip(nonzero_list_ABX, nonzero_list_base)]
intersect_abx_future = [np.intersect1d(future, abx) for future, abx in zip(nonzero_list_ABX, nonzero_list_future)]

intersect_of_intersect = [np.intersect1d(future, base) for future,
base in zip(intersect_abx_future, intersect_abx_base)]

intersect_of_intersect_counts = [np.size(inter) for inter in intersect_of_intersect]

similarity_barplots_jaccard = create_barplots(baseline_cohort, Future_cohort, "Jaccard", names, rows=7, cols=3)[0]
results_jaccard = np.vstack(create_barplots(baseline_cohort, Future_cohort, "Jaccard", names, rows=7, cols=3)[1])

baseline_cohort_no_abx_species_list = [np.delete(sample, remove
                                                 ) for sample, remove in zip(baseline_cohort, intersect_of_intersect)]
Future_cohort_no_abx_species_list = [np.delete(sample, remove
                                                 ) for sample, remove in zip(Future_cohort, intersect_of_intersect)]

similarities = [Overlap(sample_base, sample_future, overlap_type='Jaccard').calculate_overlap() for sample_base,
                sample_future in zip(baseline_cohort, Future_cohort)]
similarities_filterd = [Overlap(sample_base, sample_future, overlap_type='Jaccard'
                                ).calculate_overlap() for sample_base,
                                sample_future in zip(baseline_cohort_no_abx_species_list,
                                                     Future_cohort_no_abx_species_list)]

delta_Jaccard = [Jaccard - Jaccard_filtered for Jaccard, Jaccard_filtered in zip(similarities,
                                                                                 similarities_filterd)]

delta_Jaccard_vs_num_abx = go.Figure(data=go.Scatter(x=intersect_of_intersect_counts[14:21],
                                                     y=similarities_filterd[14:21],
                                                     mode='markers'))

delta_Jaccard_vs_num_abx.update_layout(
    xaxis={'title': 'Number of Resistant Species'},
    yaxis={'title': 'Filtered Jaccard'},
    width=500,
    height=500,
    plot_bgcolor='white'
)

# Define the coordinates for the black lines
line_coords_1 = [(-0.5, -0.5), (-0.5, 5.5)]
line_coords_2 = [(-0.5, -0.5), (5.5, -0.5)]
line_coords_3 = [(-0.5, 5.5), (5.5, 5.5)]
line_coords_4 = [(5.5, 5.5), (5.5, -0.5)]
line_coords_5 = [(5.5, 5.5), (5.5, 13.5)]
line_coords_6 = [(5.5, 5.5), (13.5, 5.5)]
line_coords_7 = [(13.5, 5.5), (13.5, 13.5)]
line_coords_8 = [(5.5, 13.5), (13.5, 13.5)]
line_coords_9 = [(13.5, 13.5), (13.5, 20.5)]
line_coords_10 = [(13.5, 13.5), (20.5, 13.5)]
line_coords_11 = [(20.5, 13.5), (20.5, 20.5)]
line_coords_12 = [(13.5, 20.5), (20.5, 20.5)]

# Create the black lines as separate traces
line_trace_1 = go.Scatter(
    x=[coord[0] for coord in line_coords_1],
    y=[coord[1] for coord in line_coords_1],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_2 = go.Scatter(
    x=[coord[0] for coord in line_coords_2],
    y=[coord[1] for coord in line_coords_2],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_3 = go.Scatter(
    x=[coord[0] for coord in line_coords_3],
    y=[coord[1] for coord in line_coords_3],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_4 = go.Scatter(
    x=[coord[0] for coord in line_coords_4],
    y=[coord[1] for coord in line_coords_4],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_5= go.Scatter(
    x=[coord[0] for coord in line_coords_5],
    y=[coord[1] for coord in line_coords_5],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_6 = go.Scatter(
    x=[coord[0] for coord in line_coords_6],
    y=[coord[1] for coord in line_coords_6],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_7 = go.Scatter(
    x=[coord[0] for coord in line_coords_7],
    y=[coord[1] for coord in line_coords_7],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_8 = go.Scatter(
    x=[coord[0] for coord in line_coords_8],
    y=[coord[1] for coord in line_coords_8],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_9 = go.Scatter(
    x=[coord[0] for coord in line_coords_9],
    y=[coord[1] for coord in line_coords_9],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_10 = go.Scatter(
    x=[coord[0] for coord in line_coords_10],
    y=[coord[1] for coord in line_coords_10],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_11 = go.Scatter(
    x=[coord[0] for coord in line_coords_11],
    y=[coord[1] for coord in line_coords_11],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

line_trace_12 = go.Scatter(
    x=[coord[0] for coord in line_coords_12],
    y=[coord[1] for coord in line_coords_12],
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='none',
    showlegend=False
)

similarity_barplots_overlap = create_barplots(baseline_cohort, Future_cohort, "Overlap", names, rows=7, cols=3)[0]
results_overlap = np.vstack(create_barplots(baseline_cohort, Future_cohort, "Overlap", names, rows=7, cols=3)[1])

# No strict

results = []

for idx, (sample_base, sample_future, ant) in enumerate(zip(baseline_cohort, Future_cohort,
                                                            ABX_cohort)):
    J_object = ShuffledVsNormal(Baseline_sample=sample_base, ABX_sample=ant, Future_sample=sample_future,
                                Baseline_cohort=baseline_cohort, index=idx, mean_num=100)
    results.append(J_object.Jaccard())

real_vals, shuffled_vals = zip(*results)
real_vals = np.array(real_vals)
shuffled_vals = np.array(shuffled_vals)

shuffled_vs_real_first_data = go.Figure()

shuffled_vs_real_first_data.add_trace(go.Scatter(x=real_vals[:6],
                                                 y=shuffled_vals[:6], mode='markers', name='aFMT',
                                                 marker=dict(color='grey')))
shuffled_vs_real_first_data.add_trace(go.Scatter(x=real_vals[6:14],
                                                 y=shuffled_vals[6:14], mode='markers', name='Probiotics',
                                                 marker=dict(color='red')))
shuffled_vs_real_first_data.add_trace(go.Scatter(x=real_vals[14:],
                                                 y=shuffled_vals[14:], mode='markers', name='Spontaneous',
                                                 marker=dict(color='blue')))

shuffled_vs_real_first_data.update_layout(
    xaxis=dict(title='Jaccard Real', title_font=dict(size=17.5), linecolor='black', linewidth=2, mirror=False),
    yaxis=dict(title='Jaccard Shuffled', title_font=dict(size=17.5), linecolor='black', linewidth=2, mirror=False),
    title=dict(text='ABX - no strict', font=dict(size=20)),
    xaxis_range=[0, 1],
    yaxis_range=[0, 1],
    shapes=[dict(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='black'))],
    width=500,
    height=500,
    legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top', font=dict(size=15)),
    plot_bgcolor='white'
)

# Strict

results = []

for idx, (sample_base, sample_future, ant) in enumerate(zip(baseline_cohort, Future_cohort,
                                                            ABX_cohort)):
    J_object = ShuffledVsNormal(Baseline_sample=sample_base, ABX_sample=ant, Future_sample=sample_future,
                                Baseline_cohort=baseline_cohort, index=idx, strict=True, mean_num=100)
    results.append(J_object.Jaccard())

real_vals, shuffled_vals = zip(*results)
real_vals = np.array(real_vals)
shuffled_vals = np.array(shuffled_vals)

shuffled_vs_real_first_data_strict = go.Figure()

shuffled_vs_real_first_data_strict.add_trace(go.Scatter(x=real_vals[:6],
                                                 y=shuffled_vals[:6], mode='markers', name='aFMT',
                                                        marker=dict(color='grey')))
shuffled_vs_real_first_data_strict.add_trace(go.Scatter(x=real_vals[6:14],
                                                 y=shuffled_vals[6:14], mode='markers', name='Probiotics',
                                                        marker=dict(color='red')))
shuffled_vs_real_first_data_strict.add_trace(go.Scatter(x=real_vals[14:],
                                                 y=shuffled_vals[14:], mode='markers', name='Spontaneous',
                                                        marker=dict(color='blue')))

shuffled_vs_real_first_data_strict.update_layout(
    xaxis=dict(title='Jaccard Real', title_font=dict(size=15), linecolor='black', linewidth=2, mirror=False),
    yaxis=dict(title='Jaccard Shuffled', title_font=dict(size=15), linecolor='black', linewidth=2, mirror=False),
    title=dict(text='ABX - no strict', font=dict(size=20)),
    xaxis_range=[0, 1],
    yaxis_range=[0, 1],
    shapes=[dict(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='black'))],
    width=500,
    height=500,
    legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top', font=dict(size=17.5)),
    plot_bgcolor='white'
)

### Recovery of gut microbiota of healthy adults following antibiotic exposure ###

rel_abund_rarefied = pd.read_csv('annotated.mOTU.rel_abund.rarefied.tsv', sep='\t')

rel_abund_rarefied = filter_data(rel_abund_rarefied)

list2 = rel_abund_rarefied['Species']
rel_abund_rarefied_data = rel_abund_rarefied.values
rel_abund_rarefied_data = rel_abund_rarefied_data[:, 1:]
rel_abund_rarefied_data = rel_abund_rarefied_data.T
rel_abund = pd.read_csv('annotated.mOTU.rel_abund.tsv',sep='\t')
rel_abund = filter_data(rel_abund)
rel_abund_HMP = pd.read_csv('annotated.mOTU.rel_abund.HMP.tsv',sep='\t')
rel_abund_HMP = filter_data(rel_abund_HMP)
rel_abund_HMP_data = rel_abund_HMP.values
rel_abund_HMP_data = rel_abund_HMP_data[:, 1:].T
rel_abund_Voigt = pd.read_csv('annotated.mOTU.rel_abund.Voigt.tsv',sep='\t')
rel_abund_Voigt = filter_data(rel_abund_Voigt)
list1 = rel_abund_Voigt['Species'].tolist()


baseline_columns = ['ERAS1_Dag0', 'ERAS2_Dag0', 'ERAS3_Dag0', 'ERAS4_Dag0', 'ERAS5_Dag0',
                    'ERAS6_Dag0', 'ERAS7_Dag0', 'ERAS8_Dag0', 'ERAS9_Dag0', 'ERAS10_Dag0',
                    'ERAS11_Dag0', 'ERAS12_Dag0']

baseline_columns_appear_4 = ['ERAS2_Dag0', 'ERAS3_Dag0', 'ERAS4_Dag0', 'ERAS5_Dag0',
                             'ERAS6_Dag0', 'ERAS7_Dag0', 'ERAS9_Dag0', 'ERAS11_Dag0', 'ERAS12_Dag0']

columns_4 = ['ERAS2_Dag4opt', 'ERAS3_Dag4', 'ERAS4_Dag4opt', 'ERAS5_Dag4', 'ERAS6_Dag4opt',
             'ERAS7_Dag4opt', 'ERAS9_Dag4', 'ERAS11_Dag4opt', 'ERAS12_Dag4opt']

columns_8 = ['ERAS1_Dag8',  'ERAS2_Dag8', 'ERAS3_Dag8', 'ERAS4_Dag8opt', 'ERAS5_Dag8', 'ERAS6_Dag8opt', 'ERAS7_Dag8',
             'ERAS8_Dag8', 'ERAS9_Dag8', 'ERAS10_Dag8', 'ERAS11_Dag8', 'ERAS12_Dag8']

columns_42 = ['ERAS1_Dag42', 'ERAS2_Dag42', 'ERAS3_Dag42', 'ERAS4_Dag42', 'ERAS5_Dag42', 'ERAS6_Dag42',
              'ERAS7_Dag42',  'ERAS8_Dag42', 'ERAS9_Dag42','ERAS10_Dag42', 'ERAS11_Dag42', 'ERAS12_Dag42']

columns_180 = ['ERAS1_Dag180', 'ERAS2_Dag180', 'ERAS3_Dag180', 'ERAS4_Dag180', 'ERAS5_Dag180', 'ERAS6_Dag180',
                'ERAS7_Dag180', 'ERAS8_Dag180', 'ERAS9_Dag180', 'ERAS10_Dag180', 'ERAS11_Dag180', 'ERAS12_Dag180']

columns_180_appear_4 = ['ERAS2_Dag180', 'ERAS3_Dag180', 'ERAS4_Dag180', 'ERAS5_Dag180', 'ERAS6_Dag180',
                        'ERAS7_Dag180', 'ERAS9_Dag180', 'ERAS11_Dag180', 'ERAS12_Dag180']

baseline_rel_abund_rarefied = rel_abund_rarefied[baseline_columns].values
baseline_rel_abund_rarefied = baseline_rel_abund_rarefied.T
baseline_rel_abund_rarefied = normalize_cohort(baseline_rel_abund_rarefied)

baseline_rel_abund_rarefied_appear_4 = rel_abund_rarefied[baseline_columns_appear_4].values
baseline_rel_abund_rarefied_appear_4 = baseline_rel_abund_rarefied_appear_4.T
baseline_rel_abund_rarefied_appear_4 = normalize_cohort(baseline_rel_abund_rarefied_appear_4)

rel_abund_rarefied_4 = rel_abund_rarefied[columns_4].values
rel_abund_rarefied_4 = rel_abund_rarefied_4.T
rel_abund_rarefied_4 = normalize_cohort(rel_abund_rarefied_4)

rel_abund_rarefied_8 = rel_abund_rarefied[columns_8].values
rel_abund_rarefied_8 = rel_abund_rarefied_8.T
rel_abund_rarefied_8 = normalize_cohort(rel_abund_rarefied_8)

rel_abund_rarefied_42 = rel_abund_rarefied[columns_42].values
rel_abund_rarefied_42 = rel_abund_rarefied_42.T
rel_abund_rarefied_42 = normalize_cohort(rel_abund_rarefied_42)

rel_abund_rarefied_180 = rel_abund_rarefied[columns_180].values
rel_abund_rarefied_180 = rel_abund_rarefied_180.T
rel_abund_rarefied_180 = normalize_cohort(rel_abund_rarefied_180)

rel_abund_rarefied_180_appear_4 = rel_abund_rarefied[columns_180_appear_4].values
rel_abund_rarefied_180_appear_4 = rel_abund_rarefied_180_appear_4.T
rel_abund_rarefied_180_appear_4 = normalize_cohort(rel_abund_rarefied_180_appear_4)

baseline_rel_abund = rel_abund[baseline_columns].values
baseline_rel_abund = baseline_rel_abund.T
baseline_rel_abund = normalize_cohort(baseline_rel_abund)

rel_abund_42 = rel_abund[columns_42].values
rel_abund_42 = rel_abund_42.T
rel_abund_42 = normalize_cohort(rel_abund_42)

rel_abund_180 = rel_abund[columns_180].values
rel_abund_180 = rel_abund_180.T
rel_abund_180 = normalize_cohort(rel_abund_180)

names = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6',
         'Subject 7', 'Subject 8', 'Subject 9', 'Subject 10', 'Subject 11', 'Subject 12']

similarity_barplots_jaccard_recovery = create_barplots(baseline_rel_abund_rarefied, rel_abund_rarefied_180,
                                                       "Jaccard", names, rows=6, cols=2)[0]
similarity_barplots_jaccard_recovery_self = create_barplots(baseline_rel_abund_rarefied, baseline_rel_abund_rarefied,
                                                            "Jaccard", names, rows=6, cols=2)[0]
similarity_barplots_jaccard_recovery_ABX = create_barplots(baseline_rel_abund_rarefied, rel_abund_rarefied_8,
                                                           "Jaccard", names, rows=6, cols=2)[0]
results_jaccard_recovery = np.vstack(create_barplots(baseline_rel_abund_rarefied, rel_abund_rarefied_180,
                                                     "Jaccard", names, rows=6, cols=2)[1])

similarity_barplots_overlap_recovery = create_barplots(baseline_rel_abund_rarefied, rel_abund_rarefied_180,
                                                       "Overlap", names, rows=6, cols=2)[0]
results_overlap_recovery = np.vstack(create_barplots(baseline_rel_abund_rarefied, rel_abund_rarefied_180,
                                                     "Overlap", names, rows=6, cols=2)[1])

nonzero_list_ABX = []
nonzero_list_ABX_counts = []
for sample in rel_abund_rarefied_8:
    nonzero_list_ABX.append(np.nonzero(sample))
    nonzero_list_ABX_counts.append(np.size(np.nonzero(sample)))
nonzero_list_base = []
nonzero_list_base_counts = []
for sample in baseline_rel_abund_rarefied:
    nonzero_list_base.append(np.nonzero(sample))
    nonzero_list_base_counts.append(np.size(np.nonzero(sample)))
nonzero_list_future = []
nonzero_list_future_counts = []
for sample in rel_abund_rarefied_180:
    nonzero_list_future.append(np.nonzero(sample))
    nonzero_list_future_counts.append(np.size(np.nonzero(sample)))

intersect = []
intersect_indexes = []

for base, future in zip(nonzero_list_base, nonzero_list_future):
    intersect.append(len(np.intersect1d(base[0], future[0])))
    intersect_indexes.append(np.intersect1d(base[0], future[0]))

intersect_frac_base = []
intersect_frac_future = []
for sample_base, sample_future, inter in zip(baseline_rel_abund_rarefied, rel_abund_rarefied_180, intersect_indexes):
    intersect_frac_base.append(np.sum(sample_base[inter]))
    intersect_frac_future.append(np.sum(sample_future[inter]))

intersect_abx_base = [np.intersect1d(base, abx) for base, abx in zip(nonzero_list_ABX, nonzero_list_base)]
intersect_abx_future = [np.intersect1d(future, abx) for future, abx in zip(nonzero_list_ABX, nonzero_list_future)]

intersect_of_intersect = [np.intersect1d(future, base) for future,
base in zip(intersect_abx_future, intersect_abx_base)]

intersect_of_intersect_counts = [np.size(inter) for inter in intersect_of_intersect]

baseline_cohort_no_abx_species_list = [np.delete(sample, remove
                                                 ) for sample, remove in zip(baseline_rel_abund_rarefied,
                                                                             intersect_of_intersect)]
Future_cohort_no_abx_species_list = [np.delete(sample, remove
                                                 ) for sample, remove in zip(rel_abund_rarefied_180,
                                                                             intersect_of_intersect)]

similarities = [Overlap(sample_base, sample_future, overlap_type='Jaccard').calculate_overlap() for sample_base,
                sample_future in zip(baseline_rel_abund_rarefied, rel_abund_rarefied_180)]
similarities_filterd = [Overlap(sample_base, sample_future, overlap_type='Jaccard'
                                ).calculate_overlap() for sample_base,
                                sample_future in zip(baseline_cohort_no_abx_species_list,
                                                     Future_cohort_no_abx_species_list)]

delta_Jaccard = [Jaccard - Jaccard_filtered for Jaccard, Jaccard_filtered in zip(similarities,
                                                                                 similarities_filterd)]

delta_Jaccard_vs_num_abx = go.Figure(data=go.Scatter(x=intersect_of_intersect_counts,
                                                     y=similarities_filterd,
                                                     mode='markers'))

delta_Jaccard_vs_num_abx.update_layout(
    xaxis={'title': 'Number of Resistant Species'},
    yaxis={'title': 'Filtered Jaccard'},
    width=500,
    height=500,
    plot_bgcolor='white'
)

# Calculate Binary Jaccard Index of each future sample from the baseline samples considering only the non ARS according
# each subject's specific ARS set.

# Strict
sets_container = []
Jaccard_sets_container = []
for base, future, abx in zip(baseline_rel_abund_rarefied, rel_abund_rarefied_180, rel_abund_rarefied_8):
    object = JaccardDisappearedSpecies(base, future, abx, strict=True)
    sets_container.append(np.setdiff1d(object.nonzero_ABX, object.intersect_of_intersect))
    Jaccard_set = []
    for sample_base in baseline_rel_abund_rarefied:
        Jaccard_set.append(object.calc_jaccard(sample_base))
    Jaccard_sets_container.append(Jaccard_set)

x = np.arange(1, len(Jaccard_sets_container)+1, 1)

similarity_barplots_dis = make_subplots(rows=6, cols=2, subplot_titles=names)

for i, result in enumerate(Jaccard_sets_container):
    row = (i // 2) + 1
    col = (i % 2) + 1
    similarity_barplots_dis .add_trace(
        go.Bar(x=np.hstack((x[0:i], x[i + 1:])), y=np.hstack((result[0:i], result[i + 1:])),
                marker_color='blue', name='All Results'), row=row, col=col)
    similarity_barplots_dis .add_trace(
        go.Bar(x=[x[i]], y=[result[i]], marker_color='red', name='Selected Result'),
        row=row, col=col)

# Update x-axis properties for all subplots
similarity_barplots_dis.update_xaxes(linecolor='black', linewidth=2, mirror=False)

# Update y-axis properties for all subplots
similarity_barplots_dis.update_yaxes(linecolor='black', linewidth=2, mirror=False)

# Update layout
similarity_barplots_dis.update_layout(height=1000, width=1200, showlegend=False, plot_bgcolor='white')

############################################
Jaccard_sets_container = []
Jaccard_vals = []
for base, abx, sets in zip(baseline_rel_abund_rarefied, rel_abund_rarefied_8, sets_container):
    mask = np.ones(len(base), dtype=bool)
    mask[sets] = False
    # Create the new vector with the complement of the indexes
    new_base = base[mask]
    new_abx = abx[mask]
    J_object = Overlap(new_base, new_abx, overlap_type='Jaccard')
    Jaccard_vals.append(J_object.calculate_overlap())

print(Jaccard_vals)
############################################

# No strict

Jaccard_sets_container_ns = []
for base, future, abx in zip(baseline_rel_abund_rarefied, rel_abund_rarefied_180, rel_abund_rarefied_8):
    object_ns = JaccardDisappearedSpecies(base, future, abx, strict=False)
    Jaccard_set_ns = []
    for sample_base in baseline_rel_abund_rarefied:
        Jaccard_set_ns.append(object_ns.calc_jaccard(sample_base))
    Jaccard_sets_container_ns.append(Jaccard_set_ns)

x = np.arange(1, len(Jaccard_sets_container_ns)+1, 1)

similarity_barplots_dis_ns = make_subplots(rows=6, cols=2, subplot_titles=names)

for i, result in enumerate(Jaccard_sets_container_ns):
    row = (i // 2) + 1
    col = (i % 2) + 1
    similarity_barplots_dis_ns.add_trace(
        go.Bar(x=np.hstack((x[0:i], x[i + 1:])), y=np.hstack((result[0:i], result[i + 1:])),
                marker_color='blue', name='All Results'), row=row, col=col)
    similarity_barplots_dis_ns.add_trace(
        go.Bar(x=[x[i]], y=[result[i]], marker_color='red', name='Selected Result'),
        row=row, col=col)

# Update x-axis properties for all subplots
similarity_barplots_dis_ns.update_xaxes(linecolor='black', linewidth=2, mirror=False)

# Update y-axis properties for all subplots
similarity_barplots_dis_ns.update_yaxes(linecolor='black', linewidth=2, mirror=False)

# Update layout
similarity_barplots_dis_ns.update_layout(height=1000, width=1200, showlegend=False, plot_bgcolor='white')

# Compare Binary Jaccard index between real and shuffled samples without including ARS species

# No strict

results = []

for idx, (sample_base, sample_future, ant) in enumerate(zip(baseline_rel_abund_rarefied, rel_abund_rarefied_180,
                                                            rel_abund_rarefied_8)):
    J_object = ShuffledVsNormal(Baseline_sample=sample_base, ABX_sample=ant, Future_sample=sample_future,
                                Baseline_cohort=baseline_rel_abund_rarefied, index=idx, mean_num=100)
    results.append(J_object.Jaccard())

real_vals, shuffled_vals = zip(*results)
real_vals = np.array(real_vals)
shuffled_vals = np.array(shuffled_vals)

shuffled_vs_real = go.Figure()

shuffled_vs_real.add_trace(go.Scatter(x=real_vals, y=shuffled_vals, mode='markers', marker=dict(color='blue')))

shuffled_vs_real.update_layout(
    xaxis=dict(title='Jaccard Real', title_font=dict(size=15), linecolor='black', linewidth=2, mirror=False),
    yaxis=dict(title='Jaccard Shuffled', title_font=dict(size=15), linecolor='black', linewidth=2, mirror=False),
    title=dict(text='ABX - no strict', font=dict(size=20)),
    xaxis_range=[0, 1],
    yaxis_range=[0, 1],
    shapes=[dict(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='black'))],
    width=500,
    height=500,
    plot_bgcolor='white'
)

# Strict

results = []

for idx, (sample_base, sample_future, ant) in enumerate(zip(baseline_rel_abund_rarefied, rel_abund_rarefied_180,
                                                            rel_abund_rarefied_8)):
    J_object = ShuffledVsNormal(Baseline_sample=sample_base, ABX_sample=ant, Future_sample=sample_future,
                                Baseline_cohort=baseline_rel_abund_rarefied, index=idx, strict=True, mean_num=100)
    results.append(J_object.Jaccard())

real_vals, shuffled_vals = zip(*results)
real_vals = np.array(real_vals)
shuffled_vals = np.array(shuffled_vals)

shuffled_vs_real_strict = go.Figure()

shuffled_vs_real_strict.add_trace(go.Scatter(x=real_vals, y=shuffled_vals, mode='markers', marker=dict(color='blue')))

shuffled_vs_real_strict.update_layout(
    xaxis=dict(title='Jaccard Real', title_font=dict(size=15), linecolor='black', linewidth=2, mirror=False),
    yaxis=dict(title='Jaccard Shuffled', title_font=dict(size=15), linecolor='black', linewidth=2, mirror=False),
    title=dict(text='ABX - strict', font=dict(size=20)),
    xaxis_range=[0, 1],
    yaxis_range=[0, 1],
    shapes=[dict(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='black'))],
    width=500,
    height=500,
    plot_bgcolor='white'
)

# PCOA

combined_data = np.vstack((baseline_rel_abund_rarefied, rel_abund_rarefied_4, rel_abund_rarefied_8,
                           rel_abund_rarefied_42, rel_abund_rarefied_180))
dist_mat = cdist(combined_data, combined_data, 'braycurtis')
mds = MDS(n_components=2, metric=True, dissimilarity='precomputed')
scaled = mds.fit_transform(dist_mat)
num_samples = np.size(baseline_rel_abund_rarefied, axis=0)
num_samples_4 = np.size(baseline_rel_abund_rarefied, axis=0)

# Create the scatter plots
PCoA_bace = go.Scatter(x=scaled[:num_samples, 0], y=scaled[:num_samples, 1], marker={"color": "blue"},
                       name='Baseline', mode="markers", text=[str(i+1) for i in range(num_samples)])
PCoA_4 = go.Scatter(x=scaled[num_samples:21, 0], y=scaled[num_samples:21, 1], marker={"color": "red"},
                    name='D4', mode="markers", text=[str(i+1) for i in range(num_samples_4)])
PCoA_8 = go.Scatter(x=scaled[21:33, 0], y=scaled[21:33, 1], marker={"color": "green"},
                    name='D8', mode="markers", text=[str(i+1) for i in range(num_samples)])
PCoA_42 = go.Scatter(x=scaled[33:45, 0], y=scaled[33:45, 1], marker={"color": "grey"},
                     name='D42', mode="markers", text=[str(i+1) for i in range(num_samples)])
PCoA_180 = go.Scatter(x=scaled[45:, 0], y=scaled[45:, 1], marker={"color": "black"},
                      name='D180', mode="markers", text=[str(i+1) for i in range(num_samples)])

ellipse_bace = confidence_ellipse(x=scaled[:num_samples, 0], y=scaled[:num_samples, 1],
                                  n_std=1.96, line=dict(color='blue'), name='Confidence Ellipse base')
ellipse_4 = confidence_ellipse(x=scaled[num_samples:21, 0], y=scaled[num_samples:21, 1],
                               n_std=1.96, line=dict(color='red'), name='Confidence Ellipse 4')
ellipse_8 = confidence_ellipse(x=scaled[21:33, 0], y=scaled[21:33, 1],
                               n_std=1.96, line=dict(color='green'), name='Confidence Ellipse 8')
ellipse_42 = confidence_ellipse(x=scaled[33:45, 0], y=scaled[33:45, 1],
                                n_std=1.96, line=dict(color='grey'), name='Confidence Ellipse 42')
ellipse_180 = confidence_ellipse(x=scaled[45:, 0], y=scaled[45:, 1],
                                 n_std=1.96, line=dict(color='black'), name='Confidence Ellipse 180')

# Create the figure and add the scatter plots and ellipses
PCoA_fig = go.Figure(data=[PCoA_bace, ellipse_bace, PCoA_4, ellipse_4, PCoA_8,
                           ellipse_8, PCoA_42, ellipse_42, PCoA_180, ellipse_180])

# Set the axis properties
PCoA_fig.update_xaxes(zeroline=False)
PCoA_fig.update_yaxes(zeroline=False)

# Update the layout
PCoA_fig.update_layout(height=750, width=750, showlegend=False, plot_bgcolor='white')

# Species barplots
mean_base = baseline_rel_abund_rarefied.mean(axis=0)
mean_ABX = rel_abund_rarefied_8.mean(axis=0)
mean_future = rel_abund_rarefied_180.mean(axis=0)

# Create the subplots
mean_barplots = make_subplots(rows=3, cols=1, shared_xaxes=True)

# Add bar traces to the subplots
mean_barplots.add_trace(go.Bar(x=np.arange(1, len(mean_base)+1, 1), y=mean_base, name='DO'), row=1, col=1)
mean_barplots.add_trace(go.Bar(x=np.arange(1, len(mean_base)+1, 1), y=mean_ABX, name='D8'), row=2, col=1)
mean_barplots.add_trace(go.Bar(x=np.arange(1, len(mean_base)+1, 1), y=mean_future, name='D180'), row=3, col=1)

# Update layout for each subplot
mean_barplots.update_xaxes(title_text='Species', row=3, col=1)

mean_barplots.update_yaxes(title_text='Mean relative abundance', row=1, col=1)
mean_barplots.update_yaxes(title_text='Mean relative abundance', row=2, col=1)
mean_barplots.update_yaxes(title_text='Mean relative abundance', row=3, col=1)

mean_barplots.update_layout(height=1100, width=1200,
                            title_text='Mean relative abundance of species', plot_bgcolor='white')

app = Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1(children='Jaccard for all species')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=similarity_barplots_jaccard)
    ]),
    #html.Div([
    #dcc.Graph(
    #        id='heatmap',
    #        figure={
    #            'data': [
    #                go.Heatmap(z=results_jaccard, colorscale='Viridis'), line_trace_1, line_trace_2, line_trace_3,
    #                line_trace_4, line_trace_5, line_trace_6, line_trace_7, line_trace_8, line_trace_9, line_trace_10,
    #                line_trace_11, line_trace_12
    #            ],
    #            'layout': go.Layout(title='Jaccard Heatmap',
    #                                xaxis={'title': 'Subjects', 'scaleanchor': 'y', 'scaleratio': 1},
    #                                yaxis={'title': 'Subjects', 'scaleanchor': 'x', 'scaleratio': 1},
    #                                width=500,
    #                                height=500,
    #                                showlegend=False)
    #        }
    #    )
    #]),
    #html.Div([
    #    html.H1(children='Overlap for all species')
    #], className='header'),
    #html.Div([
    #    dcc.Graph(figure=similarity_barplots_overlap)
    #]),
    html.Div([
        html.H1(children='Non ARS strict')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=similarity_barplots_dis_first_data),
    ]),
    html.Div([
        html.H1(children='Shuffled Jaccard compared to real Jaccard - strict')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=shuffled_vs_real_first_data_strict),
    ]),
    html.Div([
        html.H1(children='Non ARS not strict')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=similarity_barplots_dis_first_data_ns),
    ]),
    html.Div([
        html.H1(children='Shuffled Jaccard compared to real Jaccard - strict')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=shuffled_vs_real_first_data),
    ]),
    #html.Div([
    #dcc.Graph(
    #        id='heatmap',
    #        figure={
    #            'data': [
    #                go.Heatmap(z=results_overlap, colorscale='Viridis'), line_trace_1, line_trace_2, line_trace_3,
    #                line_trace_4, line_trace_5, line_trace_6, line_trace_7, line_trace_8, line_trace_9, line_trace_10,
    #                line_trace_11, line_trace_12
    #            ],
    #            'layout': go.Layout(title='Jaccard Heatmap',
    #                                xaxis={'title': 'Subjects', 'scaleanchor': 'y', 'scaleratio': 1},
    #                                yaxis={'title': 'Subjects', 'scaleanchor': 'x', 'scaleratio': 1},
    #                                width=500,
    #                                height=500,
    #                                showlegend=False)
    #        }
    #    )
    #]),
    html.Div([
        html.H1(children='Jaccard for all species - Baseline')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=similarity_barplots_jaccard_recovery_self),
    ]),
    html.Div([
        html.H1(children='Jaccard for all species')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=similarity_barplots_jaccard_recovery),
    ]),
    html.Div([
        html.H1(children='Jaccard for all species ABX vs Baseline')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=similarity_barplots_jaccard_recovery_ABX),
    ]),
    #html.Div([
    #    html.H1(children='Overlap for all species')
    #], className='header'),
    #html.Div([
    #    dcc.Graph(figure=similarity_barplots_overlap_recovery),
    #]),
    # <--------------------------------> #
    html.Div([
        html.H1(children='Non ARS strict')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=similarity_barplots_dis),
    ]),
    html.Div([
        html.H1(children='Shuffled Jaccard compared to real Jaccard - strict')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=shuffled_vs_real_strict),
    ]),
    html.Div([
        html.H1(children='Non ARS not strict')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=similarity_barplots_dis_ns),
    ]),
    html.Div([
        html.H1(children='Shuffled Jaccard compared to real Jaccard')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=shuffled_vs_real),
    ]),
    html.Div([
        html.H1(children='PCoA')
    ], className='header'),
    html.Div([
        dcc.Graph(figure=PCoA_fig),
    ]),
    #html.Div([
    #    dcc.Graph(figure=delta_Jaccard_vs_num_abx),
    #]),
    html.Div([
        dcc.Graph(figure=mean_barplots),
    ]),
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)