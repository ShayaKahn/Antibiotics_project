import os
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\Antibiotics_project')
from BC import calc_bray_curtis_dissimilarity
import pandas as pd
import numpy as np
from bray_curtis import BC
from doc import DOC
from idoa import IDOA
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_excel('Metaphlan_stool.xlsx')

# Get the sum over the columns
sum_vector = df.sum(axis=0)

# Print the sum vector
print(sum_vector)

# Auto FMT patients.
aFMT_baseline_1 = df.iloc[:, 1:8].values  # 1 to 7.
aFMT_antibiotics_1 = df.iloc[:, 8].values   # day 2 of antibiotics.
aFMT_intervention_1 = df.iloc[:, 9:19].values  # days 1,2,3,4,5,14,21,28,42,56 of intervention.
aFMT_month_1 = df.iloc[:, 19:24].values  # months after the end of intervention 1,2,3,4,6.

aFMT_baseline_2 = df.iloc[:, 24:30].values  # 1,2,3,5,6,7.
aFMT_antibiotics_2 = df.iloc[:, 30:35].values   # days 2,4,5,6,7  of antibiotics.
aFMT_intervention_2 = df.iloc[:, 35:47].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
aFMT_month_2 = df.iloc[:, 47:52].values  # months after the end of intervention 2,3,4,5,6.

aFMT_baseline_3 = df.iloc[:, 52:58].values  # 1,2,3,4,5,7.
aFMT_antibiotics_3 = df.iloc[:, 58:64].values   # days 2,3,4,5,6,7  of antibiotics.
aFMT_intervention_3 = df.iloc[:, 64:76].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
aFMT_month_3 = df.iloc[:, 76:80].values  # months after the end of intervention 2,3,4,5.

aFMT_baseline_4 = df.iloc[:, 80:86].values  # 1,3,4,5,6,7.
aFMT_antibiotics_4 = df.iloc[:, 86:91].values   # days 1,2,5,6,7  of antibiotics.
aFMT_intervention_4 = df.iloc[:, 91:102].values  # days 3,4,5,6,7,14,21,24,28,42,56 of intervention.
aFMT_month_4 = df.iloc[:, 102:106].values  # months after the end of intervention 3,4,5,6.

aFMT_baseline_5 = df.iloc[:, 106:113].values  # 1,2,3,4,5,6,7.
aFMT_antibiotics_5 = df.iloc[:, 113:119].values   # days 1,2,3,4,5,6  of antibiotics.
aFMT_intervention_5 = df.iloc[:, 119:131].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
aFMT_month_5 = df.iloc[:, 131:134].values  # months after the end of intervention 4,5,6.

aFMT_baseline_6 = df.iloc[:, 134:141].values  # 1,2,3,4,5,6,7.
aFMT_antibiotics_6 = df.iloc[:, 141:147].values   # days 2,3,4,5,6,7  of antibiotics.
aFMT_intervention_6 = df.iloc[:, 147:159].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.

# Probiotics.
pro_baseline_1 = df.iloc[:, 159:166].values  # 1 to 7.
pro_antibiotics_1 = df.iloc[:, 166:173].values   # days 1,2,3,4,5,6,7  of antibiotics.
pro_intervention_1 = df.iloc[:, 173:185].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
pro_month_1 = df.iloc[:, 185:190].values  # months after the end of intervention 2,3,4,5,6

pro_baseline_2 = df.iloc[:, 190:195].values  # 1 to 5.
pro_antibiotics_2 = df.iloc[:, 195:201].values   # days 2,3,4,5,6,7  of antibiotics.
pro_intervention_2 = df.iloc[:, 201:208].values  # days 1,2,4,5,6,14,21 of intervention.
pro_month_2 = df.iloc[:, 208:212].values  # months after the end of intervention 2,3,4,5.

pro_baseline_3 = df.iloc[:, 212:218].values  # 1,2,3,4,5,7.
pro_antibiotics_3 = df.iloc[:, 218:223].values   # days 1,2,3,4,6  of antibiotics.
pro_intervention_3 = df.iloc[:, 223:234].values  # days 1,2,3,4,5,6,14,21,28,42,56 of intervention.
pro_month_3 = df.iloc[:, 234:237].values  # months after the end of intervention 3,5,6.

pro_baseline_4 = df.iloc[:, 237:244].values  # 1 to 7.
pro_antibiotics_4 = df.iloc[:, 244:251].values   # days 1,2,3,4,5,6,7  of antibiotics.
pro_intervention_4 = df.iloc[:, 251:261].values  # days 1,2,4,5,6,14,21,28,42,56 of intervention.
pro_month_4 = df.iloc[:, 261:265].values  # months after the end of intervention 3,4,5,6.

pro_baseline_5 = df.iloc[:, 265:272].values  # 1 to 7.
pro_antibiotics_5 = df.iloc[:, 272:274].values   # days 1,2 of antibiotics.
pro_intervention_5 = df.iloc[:, 274:286].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
pro_month_5 = df.iloc[:, 286:289].values  # months after the end of intervention 4,5,6.

# Optional outlier
pro_baseline_6 = df.iloc[:, 289:296].values  # 1 to 7.
pro_antibiotics_6 = df.iloc[:, 296:302].values   # days 1,2,4,5,6,7 of antibiotics.
pro_intervention_6 = df.iloc[:, 302:314].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.

pro_baseline_7 = df.iloc[:, 314:321].values  # 1 to 7.
pro_antibiotics_7 = df.iloc[:, 321:328].values   # days 1,2,3,4,5,6,7 of antibiotics.
pro_intervention_7 = df.iloc[:, 328:340].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.

# Optional outlier
pro_baseline_8 = df.iloc[:, 340:343].values  # 1 to 3.
pro_antibiotics_8 = df.iloc[:, 343:347].values   # days 2,4,6,7 of antibiotics.
pro_intervention_8 = df.iloc[:, 347:356].values  # days 3,5,6,7,14,21,28,42,56 of intervention.

# Spontaneous.
spo_baseline_1 = df.iloc[:, 356:361].values  # 1,2,3,4,7.
spo_antibiotics_1 = df.iloc[:, 361:367].values   # days 2,3,4,5,6,7 of antibiotics.
spo_intervention_1 = df.iloc[:, 367:376].values  # days 3,4,5,7,14,21,28,42,56 of intervention.
spo_month_1 = df.iloc[:, 376:380].values  # months after the end of intervention 3,4,5,6.

spo_baseline_2 = df.iloc[:, 380:387].values  # 1,2,3,4,5,6,7.
spo_antibiotics_2 = df.iloc[:, 387:394].values   # days 1,2,3,4,5,6,7 of antibiotics.
spo_intervention_2 = df.iloc[:, 394:406].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
spo_month_2 = df.iloc[:, 406:410].values  # months after the end of intervention 3,4,5,6.

spo_baseline_3 = df.iloc[:, 410:417].values  # 1,2,3,4,5,6,7.
spo_antibiotics_3 = df.iloc[:, 417:424].values   # days 1,2,3,4,5,6,7 of antibiotics.
spo_intervention_3 = df.iloc[:, 424:436].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
spo_month_3 = df.iloc[:, 436:439].values  # months after the end of intervention 4,5,6.

spo_baseline_4 = df.iloc[:, 439:445].values  # 1,2,3,4,6,7.
spo_antibiotics_4 = df.iloc[:, 445:452].values   # days 1,2,3,4,5,6,7 of antibiotics.
spo_intervention_4 = df.iloc[:, 452:464].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.

spo_baseline_5 = df.iloc[:, 464:471].values  # 1,2,3,4,5,6,7.
spo_antibiotics_5 = df.iloc[:, 471:478].values   # days 1,2,3,4,5,6,7 of antibiotics.
spo_intervention_5 = df.iloc[:, 478:490].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.
spo_month_5 = df.iloc[:, 490:493].values  # months after the end of intervention 4,5,6.

# Optional outlier
spo_baseline_6 = df.iloc[:, 493:498].values  # 2,3,4,6,7.
spo_antibiotics_6 = df.iloc[:, 498:504].values   # days 2,3,4,5,6,7 of antibiotics.
spo_intervention_6 = df.iloc[:, 504:515].values  # days 1,2,3,4,6,7,14,21,28,42,56 of intervention.

# Optional outlier
spo_baseline_7 = df.iloc[:, 515:522].values  # 1,2,3,4,5,6,7.
spo_antibiotics_7 = df.iloc[:, 522:529].values   # days 1,2,3,4,5,6,7 of antibiotics.
spo_intervention_7 = df.iloc[:, 529:541].values  # days 1,2,3,4,5,6,7,14,21,28,42,56 of intervention.

baseline_cohort = np.vstack((aFMT_baseline_1[:, 4], aFMT_baseline_2[:, 4], aFMT_baseline_3[:, 3],
                             aFMT_baseline_4[:, 2], aFMT_baseline_5[:, 2], aFMT_baseline_6[:, 4],
                             pro_baseline_1[:, 2], pro_baseline_2[:, 2], pro_baseline_3[:, 2],
                             pro_baseline_4[:, 2], pro_baseline_5[:, 4], pro_baseline_6[:, 6],
                             pro_baseline_7[:, 2], pro_baseline_8[:, 0], spo_baseline_1[:, 2],
                             spo_baseline_2[:, 2], spo_baseline_3[:, 2], spo_baseline_4[:, 2],
                             spo_baseline_5[:, 2], spo_baseline_6[:, 3], spo_baseline_7[:, 3]))

doc_baseline = DOC(baseline_cohort)
doc_mat_baseline = doc_baseline.calc_doc()

import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
matplotlib.rcParams['text.usetex'] = True

x = doc_mat_baseline[0, :]
y = doc_mat_baseline[1, :]

# Fit a robust LOESS curve to the data
lowess = sm.nonparametric.lowess(y, x, frac=0.4)

# Plot the scatterplot and the LOESS curve
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x, y)
ax.plot(lowess[:, 0], lowess[:, 1], color='red')
ax.set_xlabel('Overlap', fontsize=15)
ax.set_ylabel('Dissimilarity', fontsize=15)
ax.set_title('DOC - baseline', fontsize=30)
plt.show()

import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm

matplotlib.rcParams['text.usetex'] = True

doc_baseline = DOC(baseline_cohort)
doc_mat_baseline = doc_baseline.calc_doc()
bs_doc_container = doc_baseline.bootstrap()

# Plot the scatterplot and the LOESS curve
fig, ax = plt.subplots(figsize=(10, 10))

x = doc_mat_baseline[0, :]
y = doc_mat_baseline[1, :]

ax.scatter(x, y)

for mat in bs_doc_container:
    x = mat[0, :]
    y = mat[1, :]

    # Fit a robust LOESS curve to the data
    lowess = sm.nonparametric.lowess(y, x, frac=0.4)

    ax.plot(lowess[:, 0], lowess[:, 1], color='red')
ax.set_xlabel('Overlap', fontsize=15)
ax.set_ylabel('Dissimilarity', fontsize=15)
ax.set_title('DOC - baseline - bootstrap', fontsize=30)
plt.show()

idoa_baseline = IDOA(baseline_cohort, baseline_cohort, identical=True, min_overlap=0.5)
idoa_baseline.calc_idoa_vector()
dissimilarity_overlap_container = idoa_baseline.dissimilarity_overlap_container

# Create a figure with 7 rows and 3 columns of subplots
figure, axs = plt.subplots(7, 3, figsize=(16, 16))

# Flatten the axs array to access each subplot individually
axs = axs.ravel()

# Loop through the subplots and plot something in each one
for i, idoa in enumerate(dissimilarity_overlap_container):
    # Scatter plot of data points
    axs[i].scatter(idoa[0, :], idoa[1, :])

    # Linear regression line
    slope, intercept = np.polyfit(idoa[0, :], idoa[1, :], 1)
    x = np.array([idoa[0, :].min(), idoa[0, :].max()])
    y = slope * x + intercept
    axs[i].plot(x, y, color='red')

    # Set axis labels and title
    axs[i].set_xlabel('Overlap', fontsize=15)
    axs[i].set_ylabel('Dissimilarity', fontsize=15)

# Add a title to the entire figure
figure.suptitle('IDOA grapgs', fontsize=25)
figure.tight_layout()

plt.show()

baseline_filtered_cohort = np.vstack((aFMT_baseline_1[:, 4], aFMT_baseline_2[:, 4], aFMT_baseline_3[:, 3],
                                      aFMT_baseline_4[:, 2], aFMT_baseline_5[:, 2], aFMT_baseline_6[:, 4],
                                      pro_baseline_1[:, 2], pro_baseline_2[:, 2], pro_baseline_3[:, 2],
                                      pro_baseline_4[:, 2], pro_baseline_5[:, 4], pro_baseline_7[:, 2],
                                      spo_baseline_1[:, 2], spo_baseline_2[:, 2], spo_baseline_3[:, 2],
                                      spo_baseline_4[:, 2], spo_baseline_5[:, 2]))

doc_filtered_baseline = DOC(baseline_filtered_cohort)
doc_mat_filtered_baseline = doc_filtered_baseline.calc_doc()

import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
matplotlib.rcParams['text.usetex'] = True

x = doc_mat_filtered_baseline[0, :]
y = doc_mat_filtered_baseline[1, :]

# Fit a robust LOESS curve to the data
lowess = sm.nonparametric.lowess(y, x, frac=0.4)

# Plot the scatterplot and the LOESS curve
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x, y)
ax.plot(lowess[:, 0], lowess[:, 1], color='red')
ax.set_xlabel('Overlap', fontsize=15)
ax.set_ylabel('Dissimilarity', fontsize=15)
ax.set_title('DOC - baseline for filtered cohort', fontsize=30)
plt.show()

import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm

matplotlib.rcParams['text.usetex'] = True

doc_filtered_baseline = DOC(baseline_filtered_cohort)
doc_mat_filtered_baseline = doc_filtered_baseline.calc_doc()
bs_doc_container_filtered = doc_filtered_baseline.bootstrap()

# Plot the scatterplot and the LOESS curve
fig, ax = plt.subplots(figsize=(10, 10))

x = doc_mat_baseline[0, :]
y = doc_mat_baseline[1, :]

ax.scatter(x, y)

for mat in bs_doc_container_filtered:
    x = mat[0, :]
    y = mat[1, :]

    # Fit a robust LOESS curve to the data
    lowess = sm.nonparametric.lowess(y, x, frac=0.4)

    ax.plot(lowess[:, 0], lowess[:, 1], color='red')
ax.set_xlabel('Overlap', fontsize=15)
ax.set_ylabel('Dissimilarity', fontsize=15)
ax.set_title('DOC - DOC - baseline for filtered cohort - bootstrap', fontsize=30)
plt.show()

idoa_filtered_baseline = IDOA(baseline_filtered_cohort, baseline_filtered_cohort, identical=True, min_overlap=0.5)
idoa_filtered_baseline.calc_idoa_vector()
dissimilarity_overlap_container_filtered = idoa_filtered_baseline.dissimilarity_overlap_container

# Create a figure with 7 rows and 3 columns of subplots
figure, axs = plt.subplots(6, 3, figsize=(16, 16))

# Flatten the axs array to access each subplot individually
axs = axs.ravel()

# Loop through the subplots and plot something in each one
for i, idoa in enumerate(dissimilarity_overlap_container_filtered):
    # Scatter plot of data points
    axs[i].scatter(idoa[0, :], idoa[1, :])

    # Linear regression line
    slope, intercept = np.polyfit(idoa[0, :], idoa[1, :], 1)
    x = np.array([idoa[0, :].min(), idoa[0, :].max()])
    y = slope * x + intercept
    axs[i].plot(x, y, color='red')

    # Set axis labels and title
    axs[i].set_xlabel('Overlap', fontsize=15)
    axs[i].set_ylabel('Dissimilarity', fontsize=15)

# Add a title to the entire figure
figure.suptitle('IDOA grapgs', fontsize=25)
figure.tight_layout()

plt.show()

# pro_baseline_6 pro_baseline_8 spo_baseline_6 spo_baseline_7

# IDOA graphs for pro_baseline_6 subject.
idoa_pro_baseline_6 = IDOA(baseline_filtered_cohort, pro_baseline_6.T, identical=False, min_overlap=0.5)
idoa_pro_baseline_6.calc_idoa_vector()
dissimilarity_overlap_container_pro_baseline_6 = idoa_pro_baseline_6.dissimilarity_overlap_container

# IDOA graphs for pro_baseline_8 subject.
idoa_pro_baseline_8 = IDOA(baseline_filtered_cohort, pro_baseline_8.T, identical=False, min_overlap=0.5)
idoa_pro_baseline_8.calc_idoa_vector()
dissimilarity_overlap_container_pro_baseline_8 = idoa_pro_baseline_8.dissimilarity_overlap_container

# IDOA graphs for spo_baseline_6 subject.
idoa_spo_baseline_6 = IDOA(baseline_filtered_cohort, spo_baseline_6.T, identical=False, min_overlap=0.5)
idoa_spo_baseline_6.calc_idoa_vector()
dissimilarity_overlap_container_spo_baseline_6 = idoa_spo_baseline_6.dissimilarity_overlap_container

# IDOA graphs for spo_baseline_8 subject.
idoa_spo_baseline_7 = IDOA(baseline_filtered_cohort, spo_baseline_7.T, identical=False, min_overlap=0.5)
idoa_spo_baseline_7.calc_idoa_vector()
dissimilarity_overlap_container_spo_baseline_7 = idoa_spo_baseline_7.dissimilarity_overlap_container

dis_over_subjects_list = [dissimilarity_overlap_container_pro_baseline_6,
                          dissimilarity_overlap_container_pro_baseline_8,
                          dissimilarity_overlap_container_spo_baseline_6,
                          dissimilarity_overlap_container_spo_baseline_7]

title_list = ['IDOA grapgs - subject 6 - probiotics', 'IDOA grapgs - subject 8 - probiotics',
              'IDOA grapgs - subject 6 - spontaneous', 'IDOA grapgs - subject 7 - spontaneous']

rows = [4, 2, 3, 4]
cols = [2, 2, 2, 2]

for subject, title, row, col in zip(dis_over_subjects_list, title_list, rows, cols):

    # Create a figure with 7 rows and 3 columns of subplots
    figure, axs = plt.subplots(row, col, figsize=(10, 10))

    # Flatten the axs array to access each subplot individually
    axs = axs.ravel()

    # Loop through the subplots and plot something in each one
    for i, idoa in enumerate(subject):
        # Scatter plot of data points
        axs[i].scatter(idoa[0, :], idoa[1, :])

        # Linear regression line
        slope, intercept = np.polyfit(idoa[0, :], idoa[1, :], 1)
        x = np.array([idoa[0, :].min(), idoa[0, :].max()])
        y = slope * x + intercept
        axs[i].plot(x, y, color='red')

        # Set axis labels and title
        axs[i].set_xlabel('Overlap', fontsize=15)
        axs[i].set_ylabel('Dissimilarity', fontsize=15)

    # Add a title to the entire figure
    figure.suptitle(title, fontsize=20)
    figure.tight_layout()

plt.show()

baseline_list = [aFMT_baseline_1, aFMT_baseline_2, aFMT_baseline_3, aFMT_baseline_4, aFMT_baseline_5, aFMT_baseline_6,
                 pro_baseline_1, pro_baseline_2, pro_baseline_3, pro_baseline_4, pro_baseline_5, pro_baseline_6,
                 pro_baseline_7, pro_baseline_8, spo_baseline_1, spo_baseline_2, spo_baseline_3, spo_baseline_4,
                 spo_baseline_5, spo_baseline_6, spo_baseline_7]

abundant_species_count = []

for subject in baseline_list:
    num_nonzero_cols = np.count_nonzero(subject, axis=0)
    abundant_species_count.append(num_nonzero_cols)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a box plot with larger size
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=abundant_species_count, ax=ax)
ax.set_title('Box Plot', fontsize=16)
ax.set_xlabel('Subject', fontsize=14)
ax.set_ylabel('Number of spicies', fontsize=14)
ax.set_xticklabels(range(1, len(abundant_species_count) + 1))
plt.show()

antibiotics_cohort = np.vstack((aFMT_antibiotics_1, aFMT_antibiotics_2[:, -1], aFMT_antibiotics_3[:, -1],
                                aFMT_antibiotics_4[:, -1], aFMT_antibiotics_5[:, -1], aFMT_antibiotics_6[:, -1],
                                pro_antibiotics_1[:, -1], pro_antibiotics_2[:, -1], pro_antibiotics_3[:, -1],
                                pro_antibiotics_4[:, -1], pro_antibiotics_5[:, -1], pro_antibiotics_6[:, -1],
                                pro_antibiotics_7[:, -1], pro_antibiotics_8[:, -1], spo_antibiotics_1[:, -1],
                                spo_antibiotics_2[:, -1], spo_antibiotics_3[:, -1], spo_antibiotics_4[:, -1],
                                spo_antibiotics_5[:, -1], spo_antibiotics_6[:, -1], spo_antibiotics_7[:, -1]))

doc_antibiotics = DOC(antibiotics_cohort)
doc_mat_antibiotics = doc_antibiotics.calc_doc()

import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
matplotlib.rcParams['text.usetex'] = True

x = doc_mat_antibiotics[0, :]
y = doc_mat_antibiotics[1, :]

# Fit a robust LOESS curve to the data
lowess = sm.nonparametric.lowess(y, x, frac=0.4)

# Plot the scatterplot and the LOESS curve
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x, y)
ax.plot(lowess[:, 0], lowess[:, 1], color='red')
ax.set_xlabel('Overlap', fontsize=15)
ax.set_ylabel('Dissimilarity', fontsize=15)
ax.set_title('DOC - antibiotics', fontsize=30)
plt.show()

import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm

matplotlib.rcParams['text.usetex'] = True

doc_antibiotics = DOC(antibiotics_cohort)
doc_mat_antibiotics = doc_antibiotics.calc_doc()
bs_doc_container_antibiotics = doc_antibiotics.bootstrap()

# Plot the scatterplot and the LOESS curve
fig, ax = plt.subplots(figsize=(10, 10))

x = doc_mat_antibiotics[0, :]
y = doc_mat_antibiotics[1, :]

ax.scatter(x, y)

for mat in bs_doc_container_antibiotics:
    x = mat[0, :]
    y = mat[1, :]

    # Fit a robust LOESS curve to the data
    lowess = sm.nonparametric.lowess(y, x, frac=0.4)

    ax.plot(lowess[:, 0], lowess[:, 1], color='red')
ax.set_xlabel('Overlap', fontsize=15)
ax.set_ylabel('Dissimilarity', fontsize=15)
ax.set_title('DOC - antibiotics', fontsize=30)
plt.show()

class Subject:
    def __init__(self, base_array=[], ant_array=[], int_array=[],
                 month_array=[], base_days=[], ant_days=[], int_days=[], month_number=[]):
        self.base_array = base_array
        self.ant_array = ant_array
        self.int_array = int_array
        self.month_array = month_array
        self.base_days = base_days
        self.ant_days = ant_days
        self.int_days = int_days
        self.month_number = month_number
    def create_base_dict(self):
        columns = [self.base_array[:, i].tolist() for i in range(self.base_array.shape[1])]
        base_dict = dict(zip(self.base_days, columns))
        return base_dict
    def create_ant_dict(self):
        if len(self.ant_days)==1:
            ant_dict = {self.ant_days[0]:self.ant_array}
            return ant_dict
        else:
            columns = [self.ant_array[:, i].tolist() for i in range(self.ant_array.shape[1])]
            ant_dict = dict(zip(self.ant_days, columns))
            return ant_dict
    def create_int_dict(self):
        columns = [self.int_array[:, i].tolist() for i in range(self.int_array.shape[1])]
        int_dict = dict(zip(self.int_days, columns))
        return int_dict
    def create_month_dict(self):
        columns = [self.month_array[:, i].tolist() for i in range(self.month_array.shape[1])]
        month_dict = dict(zip(self.month_number, columns))
        return month_dict

min_overlap_val = 0
cohort = baseline_filtered_cohort
#cohort = baseline_cohort
#cohort =antibiotics_cohort

aFMT_subject_1 = Subject(aFMT_baseline_1, aFMT_antibiotics_1, aFMT_intervention_1, aFMT_month_1,
                         ['1', '2', '3', '4', '6', '7'], ['2'],
                         ['1', '2', '3' ,'4' ,'5' ,'14' ,'21' ,'28' ,'42' ,'56'], ['1', '2', '3', '4', '6'])
aFMT_subject_1_base_dict = aFMT_subject_1.create_base_dict()
aFMT_subject_1_ant_dict = aFMT_subject_1.create_ant_dict()
aFMT_subject_1_int_dict = aFMT_subject_1.create_int_dict()
aFMT_subject_1_month_dict = aFMT_subject_1.create_month_dict()

aFMT_subject_2 = Subject(aFMT_baseline_2, aFMT_antibiotics_2, aFMT_intervention_2, aFMT_month_2,
                         ['1', '2', '3', '5', '6', '7'], ['2', '4', '5', '6', '7'],
                         ['1', '2', '3' ,'4' ,'5', '6', '7' ,'14' ,'21' ,'28' ,'42' ,'56'], ['2', '3', '4', '5', '6'])
aFMT_subject_2_base_dict = aFMT_subject_2.create_base_dict()
aFMT_subject_2_ant_dict = aFMT_subject_2.create_ant_dict()
aFMT_subject_2_int_dict = aFMT_subject_2.create_int_dict()
aFMT_subject_2_month_dict = aFMT_subject_2.create_month_dict()

aFMT_subject_3 = Subject(aFMT_baseline_3, aFMT_antibiotics_3, aFMT_intervention_3, aFMT_month_3,
                         ['1', '2', '3', '4', '5', '7'], ['2', '3', '4', '5', '6', '7'],
                         ['1', '2', '3' ,'4' ,'5', '6', '7' , '14' ,'21' ,'28' ,'42' ,'56'], ['2', '3', '4', '5'])
aFMT_subject_3_base_dict = aFMT_subject_3.create_base_dict()
aFMT_subject_3_ant_dict = aFMT_subject_3.create_ant_dict()
aFMT_subject_3_int_dict = aFMT_subject_3.create_int_dict()
aFMT_subject_3_month_dict = aFMT_subject_3.create_month_dict()

aFMT_subject_4 = Subject(aFMT_baseline_4, aFMT_antibiotics_4, aFMT_intervention_4, aFMT_month_4,
                         ['1', '3', '4', '5' ,'6' , '7'], ['1', '2', '5', '6', '7'],
                         ['3' ,'4' ,'5', '6', '7' ,'14' ,'21' ,'28' ,'42' ,'56'], ['3', '4', '5', '6'])
aFMT_subject_4_base_dict = aFMT_subject_4.create_base_dict()
aFMT_subject_4_ant_dict = aFMT_subject_4.create_ant_dict()
aFMT_subject_4_int_dict = aFMT_subject_4.create_int_dict()
aFMT_subject_4_month_dict = aFMT_subject_4.create_month_dict()

aFMT_subject_5 = Subject(aFMT_baseline_5, aFMT_antibiotics_5, aFMT_intervention_5, aFMT_month_5,
                         ['1', '2', '3', '4', '5', '6', '7'], ['1', '2', '3', '4', '5', '6'],
                         ['1', '2', '3' ,'4' ,'5', '6', '7' , '14' ,'21' ,'28' ,'42' ,'56'], ['4', '5', '6'])
aFMT_subject_5_base_dict = aFMT_subject_5.create_base_dict()
aFMT_subject_5_ant_dict = aFMT_subject_5.create_ant_dict()
aFMT_subject_5_int_dict = aFMT_subject_5.create_int_dict()
aFMT_subject_5_month_dict = aFMT_subject_5.create_month_dict()

aFMT_subject_6 = Subject(base_array=aFMT_baseline_6, ant_array=aFMT_antibiotics_6, int_array=aFMT_intervention_6,
                         base_days=['1', '2', '3', '4', '5', '6', '7'], ant_days=['2', '3', '4', '5', '6', '7'],
                         int_days=['1', '2', '3' ,'4' ,'5', '14' ,'21' ,'28' ,'42' ,'56'])
aFMT_subject_6_base_dict = aFMT_subject_6.create_base_dict()
aFMT_subject_6_ant_dict = aFMT_subject_6.create_ant_dict()
aFMT_subject_6_int_dict = aFMT_subject_6.create_int_dict()

day_1_intervension_aFMT = np.vstack((aFMT_subject_1_int_dict['1'], aFMT_subject_2_int_dict['1'],
                                     aFMT_subject_3_int_dict['1'], aFMT_subject_5_int_dict['1'],
                                     aFMT_subject_6_int_dict['1']))
idoa_1_day_intervension_aFMT = IDOA(cohort, day_1_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_1_day_intervension_aFMT_vector = idoa_1_day_intervension_aFMT.calc_idoa_vector()
idoa_1_day_intervension_aFMT_vector = idoa_1_day_intervension_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:4, 4:5})
BC_1_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_1_intervension_aFMT,
                                                                  {0:0, 1:1, 2:2, 3:4, 4:5})

day_2_intervension_aFMT = np.vstack((aFMT_subject_1_int_dict['2'], aFMT_subject_2_int_dict['2'],
                                     aFMT_subject_3_int_dict['2'], aFMT_subject_5_int_dict['2'],
                                     aFMT_subject_6_int_dict['2']))
idoa_2_day_intervension_aFMT = IDOA(cohort, day_2_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_2_day_intervension_aFMT_vector = idoa_2_day_intervension_aFMT.calc_idoa_vector()
idoa_2_day_intervension_aFMT_vector = idoa_2_day_intervension_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:4, 4:5})
BC_2_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_2_intervension_aFMT,
                                                                  {0:0, 1:1, 2:2, 3:4, 4:5})

day_3_intervension_aFMT = np.vstack((aFMT_subject_1_int_dict['3'], aFMT_subject_2_int_dict['3'],
                                     aFMT_subject_3_int_dict['3'], aFMT_subject_4_int_dict['3'],
                                     aFMT_subject_5_int_dict['3'], aFMT_subject_6_int_dict['3']))
idoa_3_day_intervension_aFMT = IDOA(cohort, day_3_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_3_day_intervension_aFMT_vector = idoa_3_day_intervension_aFMT.calc_idoa_vector()
idoa_3_day_intervension_aFMT_vector = idoa_3_day_intervension_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_3_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_3_intervension_aFMT,
                                                                  {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

day_4_intervension_aFMT = np.vstack((aFMT_subject_1_int_dict['4'], aFMT_subject_2_int_dict['4'],
                                     aFMT_subject_3_int_dict['4'], aFMT_subject_4_int_dict['4'],
                                     aFMT_subject_5_int_dict['4'], aFMT_subject_6_int_dict['4']))
idoa_4_day_intervension_aFMT = IDOA(cohort, day_4_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_4_day_intervension_aFMT_vector = idoa_4_day_intervension_aFMT.calc_idoa_vector()
idoa_4_day_intervension_aFMT_vector = idoa_4_day_intervension_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_4_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_4_intervension_aFMT,
                                                                  {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

day_5_intervension_aFMT = np.vstack((aFMT_subject_1_int_dict['5'], aFMT_subject_2_int_dict['5'],
                                     aFMT_subject_3_int_dict['5'], aFMT_subject_4_int_dict['5'],
                                     aFMT_subject_5_int_dict['5'], aFMT_subject_6_int_dict['5']))
idoa_5_day_intervension_aFMT = IDOA(cohort, day_5_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_5_day_intervension_aFMT_vector = idoa_5_day_intervension_aFMT.calc_idoa_vector()
idoa_5_day_intervension_aFMT_vector = idoa_5_day_intervension_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_5_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_5_intervension_aFMT,
                                                                  {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

day_6_intervension_aFMT = np.vstack((aFMT_subject_2_int_dict['6'], aFMT_subject_3_int_dict['6'],
                                     aFMT_subject_4_int_dict['6'], aFMT_subject_5_int_dict['6']))
idoa_6_day_intervension_aFMT = IDOA(cohort, day_6_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_6_day_intervension_aFMT_vector = idoa_6_day_intervension_aFMT.calc_idoa_vector()
idoa_6_day_intervension_aFMT_vector = idoa_6_day_intervension_aFMT.calc_idoa_vector_custom({0:1, 1:2, 2:3, 3:4})
BC_6_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_6_intervension_aFMT,
                                                                  {0:1, 1:2, 2:3, 3:4})

day_7_intervension_aFMT = np.vstack((aFMT_subject_2_int_dict['7'], aFMT_subject_3_int_dict['7'],
                                     aFMT_subject_4_int_dict['7'], aFMT_subject_5_int_dict['7']))
idoa_7_day_intervension_aFMT = IDOA(cohort, day_7_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_7_day_intervension_aFMT_vector = idoa_7_day_intervension_aFMT.calc_idoa_vector()
idoa_7_day_intervension_aFMT_vector = idoa_7_day_intervension_aFMT.calc_idoa_vector_custom({0:1, 1:2, 2:3, 3:4})
BC_7_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_7_intervension_aFMT,
                                                                  {0:1, 1:2, 2:3, 3:4})

day_14_intervension_aFMT = np.vstack((aFMT_subject_1_int_dict['14'], aFMT_subject_2_int_dict['14'],
                                      aFMT_subject_3_int_dict['14'], aFMT_subject_4_int_dict['14'],
                                      aFMT_subject_5_int_dict['14'], aFMT_subject_6_int_dict['14']))
idoa_14_day_intervension_aFMT = IDOA(cohort, day_14_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_14_day_intervension_aFMT_vector = idoa_14_day_intervension_aFMT.calc_idoa_vector()
idoa_14_day_intervension_aFMT_vector = idoa_14_day_intervension_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_14_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_14_intervension_aFMT,
                                                                   {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

day_21_intervension_aFMT = np.vstack((aFMT_subject_1_int_dict['21'], aFMT_subject_2_int_dict['21'],
                                      aFMT_subject_3_int_dict['21'], aFMT_subject_4_int_dict['21'],
                                      aFMT_subject_5_int_dict['21'], aFMT_subject_6_int_dict['21']))
idoa_21_day_intervension_aFMT = IDOA(cohort, day_21_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_21_day_intervension_aFMT_vector = idoa_21_day_intervension_aFMT.calc_idoa_vector()
idoa_21_day_intervension_aFMT_vector = idoa_21_day_intervension_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_21_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_21_intervension_aFMT,
                                                                   {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

day_28_intervension_aFMT = np.vstack((aFMT_subject_1_int_dict['28'], aFMT_subject_2_int_dict['28'],
                                      aFMT_subject_3_int_dict['28'], aFMT_subject_4_int_dict['28'],
                                      aFMT_subject_5_int_dict['28'], aFMT_subject_6_int_dict['28']))
idoa_28_day_intervension_aFMT = IDOA(cohort, day_28_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_28_day_intervension_aFMT_vector = idoa_28_day_intervension_aFMT.calc_idoa_vector()
idoa_28_day_intervension_aFMT_vector = idoa_28_day_intervension_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_28_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_28_intervension_aFMT,
                                                                   {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

day_42_intervension_aFMT = np.vstack((aFMT_subject_1_int_dict['42'], aFMT_subject_2_int_dict['42'],
                                      aFMT_subject_3_int_dict['42'], aFMT_subject_4_int_dict['42'],
                                      aFMT_subject_5_int_dict['42'], aFMT_subject_6_int_dict['42']))
idoa_42_day_intervension_aFMT = IDOA(cohort, day_42_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_42_day_intervension_aFMT_vector = idoa_42_day_intervension_aFMT.calc_idoa_vector()
idoa_42_day_intervension_aFMT_vector = idoa_42_day_intervension_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_42_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_42_intervension_aFMT,
                                                                   {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

day_56_intervension_aFMT = np.vstack((aFMT_subject_1_int_dict['56'], aFMT_subject_2_int_dict['56'],
                                      aFMT_subject_3_int_dict['56'], aFMT_subject_4_int_dict['56'],
                                      aFMT_subject_5_int_dict['56'], aFMT_subject_6_int_dict['56']))
idoa_56_day_intervension_aFMT = IDOA(cohort, day_56_intervension_aFMT, min_overlap=min_overlap_val)
#idoa_56_day_intervension_aFMT_vector = idoa_56_day_intervension_aFMT.calc_idoa_vector()
idoa_56_day_intervension_aFMT_vector = idoa_56_day_intervension_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_56_day_intervension_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_56_intervension_aFMT,
                                                                   {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

month_1_aFMT = np.vstack((aFMT_subject_1_month_dict['1']))
idoa_month_1_aFMT = IDOA(cohort, month_1_aFMT, min_overlap=min_overlap_val)
idoa_month_1_aFMT_vector = idoa_month_1_aFMT.calc_idoa_vector()
idoa_month_1_aFMT_vector = idoa_month_1_aFMT.calc_idoa_vector_custom({0:0})
BC_month_1_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, month_1_aFMT,
                                                        {0:0})

month_2_aFMT = np.vstack((aFMT_subject_1_month_dict['2'], aFMT_subject_2_month_dict['2'],
                          aFMT_subject_3_month_dict['2']))
idoa_month_2_aFMT = IDOA(cohort, month_2_aFMT, min_overlap=min_overlap_val)
#idoa_month_2_aFMT_vector = idoa_month_2_aFMT.calc_idoa_vector()
idoa_month_2_aFMT_vector = idoa_month_2_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2})
BC_month_2_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, month_2_aFMT,
                                                        {0:0, 1:1, 2:2})

month_3_aFMT = np.vstack((aFMT_subject_1_month_dict['3'], aFMT_subject_2_month_dict['3'],
                          aFMT_subject_3_month_dict['3'], aFMT_subject_4_month_dict['3']))
idoa_month_3_aFMT = IDOA(cohort, month_3_aFMT, min_overlap=min_overlap_val)
#idoa_month_3_aFMT_vector = idoa_month_3_aFMT.calc_idoa_vector()
idoa_month_3_aFMT_vector = idoa_month_3_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3})
BC_month_3_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, month_3_aFMT,
                                                        {0:0, 1:1, 2:2, 3:3})

month_4_aFMT = np.vstack((aFMT_subject_1_month_dict['4'], aFMT_subject_2_month_dict['4'],
                          aFMT_subject_3_month_dict['4'], aFMT_subject_4_month_dict['4'],
                          aFMT_subject_5_month_dict['4']))
idoa_month_4_aFMT = IDOA(cohort, month_4_aFMT, min_overlap=min_overlap_val)
#idoa_month_4_aFMT_vector = idoa_month_4_aFMT.calc_idoa_vector()
idoa_month_4_aFMT_vector = idoa_month_4_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4})
BC_month_4_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, month_4_aFMT,
                                                       {0:0, 1:1, 2:2, 3:3, 4:4})

month_5_aFMT = np.vstack((aFMT_subject_2_month_dict['5'], aFMT_subject_3_month_dict['5'],
                          aFMT_subject_4_month_dict['5'], aFMT_subject_5_month_dict['5']))
idoa_month_5_aFMT = IDOA(cohort, month_5_aFMT, min_overlap=min_overlap_val)
#idoa_month_5_aFMT_vector = idoa_month_5_aFMT.calc_idoa_vector()
idoa_month_5_aFMT_vector = idoa_month_5_aFMT.calc_idoa_vector_custom({0:1, 1:2, 2:3, 3:4})
BC_month_5_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, month_5_aFMT,
                                                       {0:1, 1:2, 2:3, 3:4})

month_6_aFMT = np.vstack((aFMT_subject_1_month_dict['6'], aFMT_subject_2_month_dict['6'],
                          aFMT_subject_4_month_dict['6'], aFMT_subject_5_month_dict['6']))
idoa_month_6_aFMT = IDOA(cohort, month_6_aFMT, min_overlap=min_overlap_val)
#idoa_month_6_aFMT_vector = idoa_month_6_aFMT.calc_idoa_vector()
idoa_month_6_aFMT_vector = idoa_month_6_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:3, 3:4})
BC_month_6_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, month_6_aFMT, {0:0, 1:1, 2:3, 3:4})

day_1_base_aFMT = np.vstack((aFMT_subject_1_base_dict['1'], aFMT_subject_2_base_dict['1'],
                             aFMT_subject_3_base_dict['1'], aFMT_subject_4_base_dict['1'],
                             aFMT_subject_5_base_dict['1']))#, aFMT_subject_6_base_dict['1']))
idoa_1_day_base_aFMT = IDOA(cohort, day_1_base_aFMT, min_overlap=min_overlap_val)
#idoa_1_day_base_aFMT_vector = idoa_1_day_base_aFMT.calc_idoa_vector()
idoa_1_day_base_aFMT_vector = idoa_1_day_base_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4})
BC_1_day_base_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_1_base_aFMT,
                                                          {0:0, 1:1, 2:2, 3:3, 4:4})

day_2_base_aFMT = np.vstack((aFMT_subject_1_base_dict['2'], aFMT_subject_2_base_dict['2'],
                             aFMT_subject_3_base_dict['2'], aFMT_subject_5_base_dict['2']))#,
                             #aFMT_subject_6_base_dict['2']))
idoa_2_day_base_aFMT = IDOA(cohort, day_2_base_aFMT, min_overlap=min_overlap_val)
#idoa_2_day_base_aFMT_vector = idoa_2_day_base_aFMT.calc_idoa_vector()
idoa_2_day_base_aFMT_vector = idoa_2_day_base_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:4})
BC_2_day_base_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_2_base_aFMT,
                                                          {0:0, 1:1, 2:2, 3:4})

day_3_base_aFMT = np.vstack((aFMT_subject_1_base_dict['3'], aFMT_subject_2_base_dict['3'],
                             aFMT_subject_3_base_dict['3'], aFMT_subject_4_base_dict['3'],
                             aFMT_subject_5_base_dict['3'], aFMT_subject_6_base_dict['3']))
idoa_3_day_base_aFMT = IDOA(cohort, day_3_base_aFMT, min_overlap=min_overlap_val)
#idoa_3_day_base_aFMT_vector = idoa_3_day_base_aFMT.calc_idoa_vector()
idoa_3_day_base_aFMT_vector = idoa_3_day_base_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_3_day_base_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_3_base_aFMT,
                                                          {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

day_4_base_aFMT = np.vstack((aFMT_subject_1_base_dict['4'], aFMT_subject_3_base_dict['4'],
                             aFMT_subject_4_base_dict['4'], aFMT_subject_5_base_dict['4'],
                             aFMT_subject_6_base_dict['4']))
idoa_4_day_base_aFMT = IDOA(cohort, day_4_base_aFMT, min_overlap=min_overlap_val)
#idoa_4_day_base_aFMT_vector = idoa_4_day_base_aFMT.calc_idoa_vector()
idoa_4_day_base_aFMT_vector = idoa_4_day_base_aFMT.calc_idoa_vector_custom({0:0, 1:2, 2:3, 3:4, 4:5})
BC_4_day_base_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_4_base_aFMT, {0:0, 1:2, 2:3, 3:4, 4:5})

day_5_base_aFMT = np.vstack((aFMT_subject_2_base_dict['5'], aFMT_subject_3_base_dict['5'],
                             aFMT_subject_4_base_dict['5'], aFMT_subject_5_base_dict['5'],
                             aFMT_subject_6_base_dict['5']))
idoa_5_day_base_aFMT = IDOA(cohort, day_5_base_aFMT, min_overlap=min_overlap_val)
#idoa_5_day_base_aFMT_vector = idoa_5_day_base_aFMT.calc_idoa_vector()
idoa_5_day_base_aFMT_vector = idoa_5_day_base_aFMT.calc_idoa_vector_custom({0:1, 1:2, 2:3, 3:4, 4:5})
BC_5_day_base_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_5_base_aFMT, {0:1, 1:2, 2:3, 3:4, 4:5})

day_6_base_aFMT = np.vstack((aFMT_subject_1_base_dict['6'], aFMT_subject_2_base_dict['6'],
                             aFMT_subject_4_base_dict['6'], aFMT_subject_5_base_dict['6'],
                             aFMT_subject_6_base_dict['6']))
idoa_6_day_base_aFMT = IDOA(cohort, day_6_base_aFMT, min_overlap=min_overlap_val)
#idoa_6_day_base_aFMT_vector = idoa_6_day_base_aFMT.calc_idoa_vector()
idoa_6_day_base_aFMT_vector = idoa_6_day_base_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:3, 3:4, 4:5})
BC_6_day_base_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_6_base_aFMT, {0:0, 1:1, 2:3, 3:4, 4:5})

day_7_base_aFMT = np.vstack((aFMT_subject_1_base_dict['7'], aFMT_subject_2_base_dict['7'],
                             aFMT_subject_3_base_dict['7'], aFMT_subject_4_base_dict['7'],
                             aFMT_subject_5_base_dict['7'], aFMT_subject_6_base_dict['7']))
idoa_7_day_base_aFMT = IDOA(cohort, day_7_base_aFMT, min_overlap=min_overlap_val)
#idoa_7_day_base_aFMT_vector = idoa_7_day_base_aFMT.calc_idoa_vector()
idoa_7_day_base_aFMT_vector = idoa_7_day_base_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_7_day_base_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_7_base_aFMT, {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

day_1_ant_aFMT = np.vstack((aFMT_subject_4_ant_dict['1'], aFMT_subject_5_ant_dict['1']))
idoa_1_day_ant_aFMT = IDOA(cohort, day_1_ant_aFMT, min_overlap=min_overlap_val)
#idoa_1_day_ant_aFMT_vector = idoa_1_day_ant_aFMT.calc_idoa_vector()
idoa_1_day_ant_aFMT_vector = idoa_1_day_ant_aFMT.calc_idoa_vector_custom({0:3, 1:4})
BC_1_day_ant_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_1_ant_aFMT, {0:3, 1:4})

day_2_ant_aFMT = np.vstack((aFMT_subject_1_ant_dict['2'], aFMT_subject_2_ant_dict['2'],
                             aFMT_subject_3_ant_dict['2'], aFMT_subject_4_ant_dict['2'],
                             aFMT_subject_5_ant_dict['2'], aFMT_subject_6_ant_dict['2']))
idoa_2_day_ant_aFMT = IDOA(cohort, day_2_ant_aFMT, min_overlap=min_overlap_val)
#idoa_2_day_ant_aFMT_vector = idoa_2_day_ant_aFMT.calc_idoa_vector()
idoa_2_day_ant_aFMT_vector = idoa_2_day_ant_aFMT.calc_idoa_vector_custom({0:0, 1:1, 2:2, 3:3, 4:4, 5:5})
BC_2_day_ant_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_2_ant_aFMT, {0:0, 1:1, 2:2, 3:3, 4:4, 5:5})

day_3_ant_aFMT = np.vstack((aFMT_subject_3_ant_dict['3'], aFMT_subject_5_ant_dict['3'],
                            aFMT_subject_6_ant_dict['3']))
idoa_3_day_ant_aFMT = IDOA(cohort, day_3_ant_aFMT, min_overlap=min_overlap_val)
#idoa_3_day_ant_aFMT_vector = idoa_3_day_ant_aFMT.calc_idoa_vector()
idoa_3_day_ant_aFMT_vector = idoa_3_day_ant_aFMT.calc_idoa_vector_custom({0:2, 1:4, 2:5})
BC_3_day_ant_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_3_ant_aFMT, {0:2, 1:4, 2:5})

day_4_ant_aFMT = np.vstack((aFMT_subject_2_ant_dict['4'], aFMT_subject_3_ant_dict['4'],
                             aFMT_subject_5_ant_dict['4'], aFMT_subject_6_ant_dict['4']))
idoa_4_day_ant_aFMT = IDOA(cohort, day_4_ant_aFMT, min_overlap=min_overlap_val)
#idoa_4_day_ant_aFMT_vector = idoa_4_day_ant_aFMT.calc_idoa_vector()
idoa_4_day_ant_aFMT_vector = idoa_4_day_ant_aFMT.calc_idoa_vector_custom({0:1, 1:2, 2:4, 3:5})
BC_4_day_ant_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_4_ant_aFMT, {0:1, 1:2, 2:4, 3:5}
)

day_5_ant_aFMT = np.vstack((aFMT_subject_2_ant_dict['5'], aFMT_subject_3_ant_dict['5'],
                             aFMT_subject_4_ant_dict['5'], aFMT_subject_5_ant_dict['5'],
                             aFMT_subject_6_ant_dict['5']))
idoa_5_day_ant_aFMT = IDOA(cohort, day_5_ant_aFMT, min_overlap=min_overlap_val)
#idoa_5_day_ant_aFMT_vector = idoa_5_day_ant_aFMT.calc_idoa_vector()
idoa_5_day_ant_aFMT_vector = idoa_5_day_ant_aFMT.calc_idoa_vector_custom({0:1, 1:2, 2:3, 3:4, 4:5})
BC_5_day_ant_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_5_ant_aFMT, {0:1, 1:2, 2:3, 3:4, 4:5})

day_6_ant_aFMT = np.vstack((aFMT_subject_2_ant_dict['6'], aFMT_subject_3_ant_dict['6'],
                             aFMT_subject_4_ant_dict['6'], aFMT_subject_5_ant_dict['6'],
                             aFMT_subject_6_ant_dict['6']))
idoa_6_day_ant_aFMT = IDOA(cohort, day_6_ant_aFMT, min_overlap=min_overlap_val)
#idoa_6_day_ant_aFMT_vector = idoa_6_day_ant_aFMT.calc_idoa_vector()
idoa_6_day_ant_aFMT_vector = idoa_6_day_ant_aFMT.calc_idoa_vector_custom({0:1, 1:2, 2:3, 3:4, 4:5})
BC_6_day_ant_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_6_ant_aFMT, {0:1, 1:2, 2:3, 3:4, 4:5})

day_7_ant_aFMT = np.vstack((aFMT_subject_2_ant_dict['7'], aFMT_subject_3_ant_dict['7'],
                             aFMT_subject_4_ant_dict['7'], aFMT_subject_6_ant_dict['7']))
idoa_7_day_ant_aFMT = IDOA(cohort, day_7_ant_aFMT, min_overlap=min_overlap_val)
#idoa_7_day_ant_aFMT_vector = idoa_7_day_ant_aFMT.calc_idoa_vector()
idoa_7_day_ant_aFMT_vector = idoa_7_day_ant_aFMT.calc_idoa_vector_custom({0:1, 1:2, 2:3, 3:5})
BC_7_day_ant_aFMT_vector = calc_bray_curtis_dissimilarity(cohort, day_7_ant_aFMT, {0:1, 1:2, 2:3, 3:5})

pro_subject_1 = Subject(pro_baseline_1, pro_antibiotics_1 ,pro_intervention_1, pro_month_1,
                        ['1', '2', '3', '4', '5', '6', '7'], ['1', '2', '3', '4', '5', '6', '7'],
                        ['1', '2', '3' ,'4' ,'5', '6', '7' ,'14' ,'21' ,'28' ,'42' ,'56'], ['2', '3', '4', '5', '6'])
pro_subject_1_base_dict = pro_subject_1.create_base_dict()
pro_subject_1_ant_dict = pro_subject_1.create_ant_dict()
pro_subject_1_int_dict = pro_subject_1.create_int_dict()
pro_subject_1_month_dict = pro_subject_1.create_month_dict()

pro_subject_2 = Subject(pro_baseline_2, pro_antibiotics_2, pro_intervention_2, pro_month_2,
                        ['1', '2', '3', '4', '5'], ['2', '3', '4', '5', '6', '7'],
                        ['1', '2', '4', '5', '6', '14', '21'], ['2', '3', '4', '5'])
pro_subject_2_base_dict = pro_subject_2.create_base_dict()
pro_subject_2_ant_dict = pro_subject_2.create_ant_dict()
pro_subject_2_int_dict = pro_subject_2.create_int_dict()
pro_subject_2_month_dict = pro_subject_2.create_month_dict()

pro_subject_3 = Subject(pro_baseline_3, pro_antibiotics_3, pro_intervention_3, pro_month_3,
                        ['1', '2', '3', '4', '5', '7'], ['1', '2', '3', '4', '6'],
                        ['1', '2', '3', '4', '5', '6', '14', '21', '28', '42', '56'], ['3', '5', '6'])
pro_subject_3_base_dict = pro_subject_1.create_base_dict()
pro_subject_3_ant_dict = pro_subject_1.create_ant_dict()
pro_subject_3_int_dict = pro_subject_3.create_int_dict()
pro_subject_3_month_dict = pro_subject_3.create_month_dict()

pro_subject_4 = Subject(pro_baseline_4, pro_antibiotics_4, pro_intervention_4, pro_month_4,
                        ['1', '2', '3', '4', '5', '6', '7'], ['1', '2', '3', '4', '5', '6', '7'],
                        ['1', '2', '4', '5', '6', '14', '21', '28', '42', '56'], ['3', '4', '5', '6'])
pro_subject_4_base_dict = pro_subject_4.create_base_dict()
pro_subject_4_ant_dict = pro_subject_4.create_ant_dict()
pro_subject_4_int_dict = pro_subject_4.create_int_dict()
pro_subject_4_month_dict = pro_subject_4.create_month_dict()

pro_subject_5 = Subject(pro_baseline_5, pro_antibiotics_5, pro_intervention_5, pro_month_5,
                        ['1', '2', '3', '4', '5', '6', '7'], ['1', '2'],
                        ['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'], ['4', '5', '6'])
pro_subject_5_base_dict = pro_subject_5.create_base_dict()
pro_subject_5_ant_dict = pro_subject_5.create_ant_dict()
pro_subject_5_int_dict = pro_subject_5.create_int_dict()
pro_subject_5_month_dict = pro_subject_5.create_month_dict()

#pro_subject_6 = Subject(int_array=pro_intervention_6, int_days=['1', '2', '3', '4', '5', '6',
                                                                #'7', '14', '21', '28', '42', '56'])
#pro_subject_6_int_dict = pro_subject_6.create_int_dict()

pro_subject_7 = Subject(base_array=pro_baseline_7, ant_array=pro_antibiotics_7, int_array=pro_intervention_7,
                        base_days=['1', '2', '3', '4', '5', '6', '7'], ant_days=['1', '2', '3', '4', '5', '6', '7'],
                        int_days=['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'])
pro_subject_7_base_dict = pro_subject_7.create_base_dict()
pro_subject_7_ant_dict = pro_subject_7.create_ant_dict()
pro_subject_7_int_dict = pro_subject_7.create_int_dict()

#pro_subject_8 = Subject(int_array=pro_intervention_8, int_days=['3', '5', '6', '7', '14', '21', '28', '42', '56'])
#pro_subject_8_int_dict = pro_subject_8.create_int_dict()

day_1_intervension_pro = np.vstack((pro_subject_1_int_dict['1'], pro_subject_2_int_dict['1'],
                                     pro_subject_3_int_dict['1'], pro_subject_5_int_dict['1'],
                                     #pro_subject_6_int_dict['1'],
                                     pro_subject_7_int_dict['1']))
idoa_1_day_intervension_pro = IDOA(cohort, day_1_intervension_pro, min_overlap=min_overlap_val)
#idoa_1_day_intervension_pro_vector = idoa_1_day_intervension_pro.calc_idoa_vector()
idoa_1_day_intervension_pro_vector = idoa_1_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:10, 4:11})
BC_1_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_1_intervension_pro,
                                                                  {0:6, 1:7, 2:8, 3:10, 4:11})

day_2_intervension_pro = np.vstack((pro_subject_1_int_dict['2'], pro_subject_2_int_dict['2'],
                                     pro_subject_3_int_dict['2'], pro_subject_5_int_dict['2'],
                                     #pro_subject_6_int_dict['2'],
                                     pro_subject_7_int_dict['2']))
idoa_2_day_intervension_pro = IDOA(cohort, day_2_intervension_pro, min_overlap=min_overlap_val)
#idoa_2_day_intervension_pro_vector = idoa_2_day_intervension_pro.calc_idoa_vector()
idoa_2_day_intervension_pro_vector = idoa_2_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:10, 4:11})
BC_2_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_2_intervension_pro,
                                                                  {0:6, 1:7, 2:8, 3:10, 4:11})

day_3_intervension_pro = np.vstack((pro_subject_1_int_dict['3'], pro_subject_3_int_dict['3'],
                                    pro_subject_5_int_dict['3'], #pro_subject_6_int_dict['3'],
                                    pro_subject_7_int_dict['3']))#, pro_subject_8_int_dict['3']))
idoa_3_day_intervension_pro = IDOA(cohort, day_3_intervension_pro, min_overlap=min_overlap_val)
#idoa_3_day_intervension_pro_vector = idoa_3_day_intervension_pro.calc_idoa_vector()
idoa_3_day_intervension_pro_vector = idoa_3_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:8, 2:10, 3:11})
BC_3_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_3_intervension_pro, {0:6, 1:8, 2:10, 3:11})

day_4_intervension_pro = np.vstack((pro_subject_1_int_dict['4'], pro_subject_2_int_dict['4'],
                                     pro_subject_3_int_dict['4'], pro_subject_4_int_dict['4'],
                                     pro_subject_5_int_dict['4'], #pro_subject_6_int_dict['4'],
                                     pro_subject_7_int_dict['4']))
idoa_4_day_intervension_pro = IDOA(cohort, day_4_intervension_pro, min_overlap=min_overlap_val)
#idoa_4_day_intervension_pro_vector = idoa_4_day_intervension_pro.calc_idoa_vector()
idoa_4_day_intervension_pro_vector = idoa_4_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_4_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_4_intervension_pro,
                                                                  {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_5_intervension_pro = np.vstack((pro_subject_1_int_dict['5'], pro_subject_2_int_dict['5'],
                                     pro_subject_3_int_dict['5'], pro_subject_4_int_dict['5'],
                                     pro_subject_5_int_dict['5'], #pro_subject_6_int_dict['5'],
                                     pro_subject_7_int_dict['5']))#, pro_subject_8_int_dict['5']))
idoa_5_day_intervension_pro = IDOA(cohort, day_5_intervension_pro, min_overlap=min_overlap_val)
#idoa_5_day_intervension_pro_vector = idoa_5_day_intervension_pro.calc_idoa_vector()
idoa_5_day_intervension_pro_vector = idoa_5_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_5_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_5_intervension_pro,
                                                                  {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_6_intervension_pro = np.vstack((pro_subject_1_int_dict['6'], pro_subject_2_int_dict['6'],
                                     pro_subject_3_int_dict['6'], pro_subject_4_int_dict['6'],
                                     pro_subject_5_int_dict['6'], #pro_subject_6_int_dict['6'],
                                     pro_subject_7_int_dict['6']))#pro_subject_8_int_dict['6']))
idoa_6_day_intervension_pro = IDOA(cohort, day_6_intervension_pro, min_overlap=min_overlap_val)
#idoa_6_day_intervension_pro_vector = idoa_6_day_intervension_pro.calc_idoa_vector()
idoa_6_day_intervension_pro_vector = idoa_6_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_6_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_6_intervension_pro,
                                                                  {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_7_intervension_pro = np.vstack((pro_subject_1_int_dict['7'], pro_subject_5_int_dict['7'],
                                    #pro_subject_6_int_dict['7'],
                                    pro_subject_7_int_dict['7'],
                                    ))#pro_subject_8_int_dict['7']))
idoa_7_day_intervension_pro = IDOA(cohort, day_7_intervension_pro, min_overlap=min_overlap_val)
#idoa_7_day_intervension_pro_vector = idoa_7_day_intervension_pro.calc_idoa_vector()
idoa_7_day_intervension_pro_vector = idoa_7_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:10, 2:11})
BC_7_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_7_intervension_pro, {0:6, 1:10, 2:11})

day_14_intervension_pro = np.vstack((pro_subject_1_int_dict['14'], pro_subject_2_int_dict['14'],
                                      pro_subject_3_int_dict['14'], pro_subject_4_int_dict['14'],
                                      pro_subject_5_int_dict['14'], #pro_subject_6_int_dict['14'],
                                      pro_subject_7_int_dict['14'], ))#pro_subject_8_int_dict['14']))
idoa_14_day_intervension_pro = IDOA(cohort, day_14_intervension_pro, min_overlap=min_overlap_val)
#idoa_14_day_intervension_pro_vector = idoa_14_day_intervension_pro.calc_idoa_vector()
idoa_14_day_intervension_pro_vector = idoa_14_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_14_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_14_intervension_pro,
                                                                   {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_21_intervension_pro = np.vstack((pro_subject_1_int_dict['21'], pro_subject_2_int_dict['21'],
                                      pro_subject_3_int_dict['21'], pro_subject_4_int_dict['21'],
                                      pro_subject_5_int_dict['21'], #pro_subject_6_int_dict['21'],
                                      pro_subject_7_int_dict['21'],
                                      ))#pro_subject_8_int_dict['21']))
idoa_21_day_intervension_pro = IDOA(cohort, day_21_intervension_pro, min_overlap=min_overlap_val)
#idoa_21_day_intervension_pro_vector = idoa_21_day_intervension_pro.calc_idoa_vector()
idoa_21_day_intervension_pro_vector = idoa_21_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_21_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_21_intervension_pro,
                                                                  {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_28_intervension_pro = np.vstack((pro_subject_1_int_dict['28'], pro_subject_3_int_dict['28'],
                                     pro_subject_4_int_dict['28'], pro_subject_5_int_dict['28'],
                                     #pro_subject_6_int_dict['28'],
                                     pro_subject_7_int_dict['28'],
                                    ))#pro_subject_8_int_dict['28']))
idoa_28_day_intervension_pro = IDOA(cohort, day_28_intervension_pro, min_overlap=min_overlap_val)
#idoa_28_day_intervension_pro_vector = idoa_28_day_intervension_pro.calc_idoa_vector()
idoa_28_day_intervension_pro_vector = idoa_28_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:8, 2:9, 3:10, 4:11})
BC_28_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_28_intervension_pro,
                                                                  {0:6, 1:8, 2:9, 3:10, 4:11})

day_42_intervension_pro = np.vstack((pro_subject_1_int_dict['42'], pro_subject_3_int_dict['42'],
                                     pro_subject_4_int_dict['42'], pro_subject_5_int_dict['42'],
                                     #pro_subject_6_int_dict['42'],
                                     pro_subject_7_int_dict['42'],
                                    ))#pro_subject_8_int_dict['42']))
idoa_42_day_intervension_pro = IDOA(cohort, day_42_intervension_pro, min_overlap=min_overlap_val)
#idoa_42_day_intervension_pro_vector = idoa_42_day_intervension_pro.calc_idoa_vector()
idoa_42_day_intervension_pro_vector = idoa_42_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:8, 2:9, 3:10, 4:11})
BC_42_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_42_intervension_pro,
                                                                   {0:6, 1:8, 2:9, 3:10, 4:11})

day_56_intervension_pro = np.vstack((pro_subject_1_int_dict['56'], pro_subject_3_int_dict['56'],
                                     pro_subject_4_int_dict['56'], pro_subject_5_int_dict['56'],
                                     #pro_subject_6_int_dict['56'],
                                     pro_subject_7_int_dict['56'],
                                     ))#pro_subject_8_int_dict['56']))
idoa_56_day_intervension_pro = IDOA(cohort, day_56_intervension_pro, min_overlap=min_overlap_val)
#idoa_56_day_intervension_pro_vector = idoa_56_day_intervension_pro.calc_idoa_vector()
idoa_56_day_intervension_pro_vector = idoa_56_day_intervension_pro.calc_idoa_vector_custom({0:6, 1:8, 2:9, 3:10, 4:11})
BC_56_day_intervension_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_56_intervension_pro,
                                                                   {0:6, 1:8, 2:9, 3:10, 4:11})

month_2_pro = np.vstack((pro_subject_1_month_dict['2'], pro_subject_2_month_dict['2']))
idoa_month_2_pro = IDOA(cohort, month_2_pro, min_overlap=min_overlap_val)
#idoa_month_2_pro_vector = idoa_month_2_pro.calc_idoa_vector()
idoa_month_2_pro_vector = idoa_month_2_pro.calc_idoa_vector_custom({0:6, 1:7})
BC_month_2_pro_vector = calc_bray_curtis_dissimilarity(cohort, month_2_pro, {0:6, 1:7})

month_3_pro = np.vstack((pro_subject_1_month_dict['3'], pro_subject_2_month_dict['3'],
                          pro_subject_3_month_dict['3'], pro_subject_4_month_dict['3']))
idoa_month_3_pro = IDOA(cohort, month_3_pro, min_overlap=min_overlap_val)
#idoa_month_3_pro_vector = idoa_month_3_pro.calc_idoa_vector()
idoa_month_3_pro_vector = idoa_month_3_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9})
BC_month_3_pro_vector = calc_bray_curtis_dissimilarity(cohort, month_3_pro, {0:6, 1:7, 2:8, 3:9})

month_4_pro = np.vstack((pro_subject_1_month_dict['4'], pro_subject_2_month_dict['4'],
                         pro_subject_4_month_dict['4'], pro_subject_5_month_dict['4']))
idoa_month_4_pro = IDOA(cohort, month_4_pro, min_overlap=min_overlap_val)
#idoa_month_4_pro_vector = idoa_month_4_pro.calc_idoa_vector()
idoa_month_4_pro_vector = idoa_month_4_pro.calc_idoa_vector_custom({0:6, 1:7, 2:9, 3:10})
BC_month_4_pro_vector = calc_bray_curtis_dissimilarity(cohort, month_4_pro, {0:6, 1:7, 2:9, 3:10})

month_5_pro = np.vstack((pro_subject_1_month_dict['5'], pro_subject_2_month_dict['5'],
                         pro_subject_3_month_dict['5'], pro_subject_4_month_dict['5'],
                         pro_subject_5_month_dict['5']))
idoa_month_5_pro = IDOA(cohort, month_5_pro, min_overlap=min_overlap_val)
#idoa_month_5_pro_vector = idoa_month_5_pro.calc_idoa_vector()
idoa_month_5_pro_vector = idoa_month_5_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10})
BC_month_5_pro_vector = calc_bray_curtis_dissimilarity(cohort, month_5_pro, {0:6, 1:7, 2:8, 3:9, 4:10})

month_6_pro = np.vstack((pro_subject_1_month_dict['6'], pro_subject_3_month_dict['6'],
                         pro_subject_4_month_dict['6'], pro_subject_5_month_dict['6']))
idoa_month_6_pro = IDOA(cohort, month_6_pro, min_overlap=min_overlap_val)
#idoa_month_6_pro_vector = idoa_month_6_pro.calc_idoa_vector()
idoa_month_6_pro_vector = idoa_month_6_pro.calc_idoa_vector_custom({0:6, 1:8, 2:9, 3:10})
BC_month_6_pro_vector = calc_bray_curtis_dissimilarity(cohort, month_6_pro, {0:6, 1:8, 2:9, 3:10})

day_1_base_pro = np.vstack((pro_subject_1_base_dict['1'], pro_subject_2_base_dict['1'],
                             pro_subject_3_base_dict['1'], pro_subject_4_base_dict['1'],
                             pro_subject_5_base_dict['1'], pro_subject_7_base_dict['1']))
idoa_1_day_base_pro = IDOA(cohort, day_1_base_pro, min_overlap=min_overlap_val)
#idoa_1_day_base_pro_vector = idoa_1_day_base_pro.calc_idoa_vector()
idoa_1_day_base_pro_vector = idoa_1_day_base_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_1_day_base_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_1_base_pro, {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_2_base_pro = np.vstack((pro_subject_1_base_dict['2'], pro_subject_2_base_dict['2'],
                            pro_subject_3_base_dict['2'], pro_subject_4_base_dict['2'],
                            pro_subject_5_base_dict['2'], pro_subject_7_base_dict['2']))
idoa_2_day_base_pro = IDOA(cohort, day_2_base_pro, min_overlap=min_overlap_val)
#doa_2_day_base_pro_vector = idoa_2_day_base_pro.calc_idoa_vector()
idoa_2_day_base_pro_vector = idoa_2_day_base_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_2_day_base_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_2_base_pro, {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_3_base_pro = np.vstack((pro_subject_1_base_dict['3'], pro_subject_2_base_dict['3'],
                            pro_subject_3_base_dict['3'], pro_subject_4_base_dict['3'],
                            pro_subject_5_base_dict['3'], pro_subject_7_base_dict['3']))
idoa_3_day_base_pro = IDOA(cohort, day_3_base_pro, min_overlap=min_overlap_val)
#idoa_3_day_base_pro_vector = idoa_3_day_base_pro.calc_idoa_vector()
idoa_3_day_base_pro_vector = idoa_3_day_base_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_3_day_base_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_3_base_pro, {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_4_base_pro = np.vstack((pro_subject_1_base_dict['4'], pro_subject_2_base_dict['4'],
                            pro_subject_3_base_dict['4'], pro_subject_4_base_dict['4'],
                            pro_subject_5_base_dict['4'], pro_subject_7_base_dict['4']))
idoa_4_day_base_pro = IDOA(cohort, day_4_base_pro, min_overlap=min_overlap_val)
#idoa_4_day_base_pro_vector = idoa_4_day_base_pro.calc_idoa_vector()
idoa_4_day_base_pro_vector = idoa_4_day_base_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_4_day_base_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_4_base_pro, {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_5_base_pro = np.vstack((pro_subject_1_base_dict['5'], pro_subject_2_base_dict['5'],
                            pro_subject_3_base_dict['5'], pro_subject_4_base_dict['5'],
                            pro_subject_5_base_dict['5'], pro_subject_7_base_dict['5']))
idoa_5_day_base_pro = IDOA(cohort, day_5_base_pro, min_overlap=min_overlap_val)
#idoa_5_day_base_pro_vector = idoa_5_day_base_pro.calc_idoa_vector()
idoa_5_day_base_pro_vector = idoa_5_day_base_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_5_day_base_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_5_base_pro, {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_6_base_pro = np.vstack((pro_subject_1_base_dict['6'], pro_subject_4_base_dict['6'],
                            pro_subject_5_base_dict['6'], pro_subject_7_base_dict['6']))
idoa_6_day_base_pro = IDOA(cohort, day_6_base_pro, min_overlap=min_overlap_val)
#idoa_6_day_base_pro_vector = idoa_6_day_base_pro.calc_idoa_vector()
idoa_6_day_base_pro_vector = idoa_6_day_base_pro.calc_idoa_vector_custom({0:6, 1:9, 2:10, 3:11})
BC_6_day_base_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_6_base_pro, {0:6, 1:9, 2:10, 3:11})

day_7_base_pro = np.vstack((pro_subject_1_base_dict['7'], pro_subject_3_base_dict['7'],
                            pro_subject_4_base_dict['7'], pro_subject_5_base_dict['7'],
                            pro_subject_7_base_dict['7']))
idoa_7_day_base_pro = IDOA(cohort, day_7_base_pro, min_overlap=min_overlap_val)
#idoa_7_day_base_pro_vector = idoa_7_day_base_pro.calc_idoa_vector()
idoa_7_day_base_pro_vector = idoa_7_day_base_pro.calc_idoa_vector_custom({0:6, 1:8, 2:9, 3:10, 4:11})
BC_7_day_base_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_7_base_pro, {0:6, 1:8, 2:9, 3:10, 4:11})

day_1_ant_pro = np.vstack((pro_subject_1_ant_dict['1'], pro_subject_3_ant_dict['1'],
                           pro_subject_4_ant_dict['1'], pro_subject_5_ant_dict['1'],
                           pro_subject_7_ant_dict['1']))
idoa_1_day_ant_pro = IDOA(cohort, day_1_ant_pro, min_overlap=min_overlap_val)
#idoa_1_day_ant_pro_vector = idoa_1_day_ant_pro.calc_idoa_vector()
idoa_1_day_ant_pro_vector = idoa_1_day_ant_pro.calc_idoa_vector_custom({0:6, 1:8, 2:9, 3:10, 4:11})
BC_1_day_ant_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_1_ant_pro, {0:6, 1:8, 2:9, 3:10, 4:11})

day_2_ant_pro = np.vstack((pro_subject_1_ant_dict['2'], pro_subject_2_ant_dict['2'],
                           pro_subject_3_ant_dict['2'], pro_subject_4_ant_dict['2'],
                           pro_subject_5_ant_dict['2'], pro_subject_7_ant_dict['2']))
idoa_2_day_ant_pro = IDOA(cohort, day_2_ant_pro, min_overlap=min_overlap_val)
#idoa_2_day_ant_pro_vector = idoa_2_day_ant_pro.calc_idoa_vector()
idoa_2_day_ant_pro_vector = idoa_2_day_ant_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:10, 5:11})
BC_2_day_ant_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_2_ant_pro, {0:6, 1:7, 2:8, 3:9, 4:10, 5:11})

day_3_ant_pro = np.vstack((pro_subject_1_ant_dict['3'], pro_subject_2_ant_dict['3'],
                           pro_subject_3_ant_dict['3'], pro_subject_4_ant_dict['3'],
                           pro_subject_7_ant_dict['3']))
idoa_3_day_ant_pro = IDOA(cohort, day_3_ant_pro, min_overlap=min_overlap_val)
#idoa_3_day_ant_pro_vector = idoa_3_day_ant_pro.calc_idoa_vector()
idoa_3_day_ant_pro_vector = idoa_3_day_ant_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:11})
BC_3_day_ant_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_3_ant_pro, {0:6, 1:7, 2:8, 3:9, 4:11})

day_4_ant_pro = np.vstack((pro_subject_1_ant_dict['4'], pro_subject_2_ant_dict['4'],
                           pro_subject_3_ant_dict['4'], pro_subject_4_ant_dict['4'],
                           pro_subject_7_ant_dict['4']))
idoa_4_day_ant_pro = IDOA(cohort, day_4_ant_pro, min_overlap=min_overlap_val)
#idoa_4_day_ant_pro_vector = idoa_4_day_ant_pro.calc_idoa_vector()
idoa_4_day_ant_pro_vector = idoa_4_day_ant_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:11})
BC_4_day_ant_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_4_ant_pro, {0:6, 1:7, 2:8, 3:9, 4:11})

day_5_ant_pro = np.vstack((pro_subject_1_ant_dict['5'], pro_subject_2_ant_dict['5'],
                           pro_subject_4_ant_dict['5'], pro_subject_7_ant_dict['5']))
idoa_5_day_ant_pro = IDOA(cohort, day_5_ant_pro, min_overlap=min_overlap_val)
#idoa_5_day_ant_pro_vector = idoa_5_day_ant_pro.calc_idoa_vector()
idoa_5_day_ant_pro_vector = idoa_5_day_ant_pro.calc_idoa_vector_custom({0:6, 1:7, 2:9, 3:11})
BC_5_day_ant_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_5_ant_pro, {0:6, 1:7, 2:9, 3:11})

day_6_ant_pro = np.vstack((pro_subject_1_ant_dict['6'], pro_subject_2_ant_dict['6'],
                           pro_subject_3_ant_dict['6'], pro_subject_4_ant_dict['6'],
                          pro_subject_7_ant_dict['6']))
idoa_6_day_ant_pro = IDOA(cohort, day_6_ant_pro, min_overlap=min_overlap_val)
#idoa_6_day_ant_pro_vector = idoa_6_day_ant_pro.calc_idoa_vector()
idoa_6_day_ant_pro_vector = idoa_6_day_ant_pro.calc_idoa_vector_custom({0:6, 1:7, 2:8, 3:9, 4:11})
BC_6_day_ant_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_6_ant_pro, {0:6, 1:7, 2:8, 3:9, 4:11})

day_7_ant_pro = np.vstack((pro_subject_1_ant_dict['7'], pro_subject_2_ant_dict['7'],
                           pro_subject_4_ant_dict['7'], pro_subject_7_ant_dict['7']))
idoa_7_day_ant_pro = IDOA(cohort, day_7_ant_pro, min_overlap=min_overlap_val)
#idoa_7_day_ant_pro_vector = idoa_7_day_ant_pro.calc_idoa_vector()
idoa_7_day_ant_pro_vector = idoa_7_day_ant_pro.calc_idoa_vector_custom({0:6, 1:7, 2:9, 3:11})
BC_7_day_ant_pro_vector = calc_bray_curtis_dissimilarity(cohort, day_7_ant_pro, {0:6, 1:7, 2:9, 3:11})

spo_subject_1 = Subject(spo_baseline_1, spo_antibiotics_1, spo_intervention_1, spo_month_1,
                        ['1', '2', '3', '4', '7'], ['2', '3', '4', '5', '6', '7'],
                        ['3', '4', '5', '7', '14', '21', '28', '42', '56'], ['3', '4', '5', '6'])
spo_subject_1_base_dict = spo_subject_1.create_base_dict()
spo_subject_1_ant_dict = spo_subject_1.create_ant_dict()
spo_subject_1_int_dict = spo_subject_1.create_int_dict()
spo_subject_1_month_dict = spo_subject_1.create_month_dict()

spo_subject_2 = Subject(spo_baseline_2, spo_antibiotics_2, spo_intervention_2, spo_month_2,
                        ['1', '2', '3', '4', '5', '6', '7'], ['1', '2', '3', '4', '5', '6', '7'],
                        ['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'], ['3', '4', '5', '6'])
spo_subject_2_base_dict = spo_subject_2.create_base_dict()
spo_subject_2_ant_dict = spo_subject_2.create_ant_dict()
spo_subject_2_int_dict = spo_subject_2.create_int_dict()
spo_subject_2_month_dict = spo_subject_2.create_month_dict()

spo_subject_3 = Subject(spo_baseline_3, spo_antibiotics_3, spo_intervention_3, spo_month_3,
                        ['1', '2', '3', '4', '5', '6', '7'], ['1', '2', '3', '4', '5', '6', '7'],
                        ['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'], ['4', '5', '6'])
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

spo_subject_5 = Subject(spo_baseline_5, spo_antibiotics_5, spo_intervention_5, spo_month_5,
                        ['1', '2', '3', '4', '5', '6', '7'], ['1', '2', '3', '4', '5', '6', '7'],
                        ['1', '2', '3', '4', '5', '6', '7', '14', '21', '28', '42', '56'], ['4', '5', '6'])
spo_subject_5_base_dict = spo_subject_5.create_base_dict()
spo_subject_5_ant_dict = spo_subject_5.create_ant_dict()
spo_subject_5_int_dict = spo_subject_5.create_int_dict()
spo_subject_5_month_dict = spo_subject_5.create_month_dict()

day_1_intervension_spo = np.vstack((spo_subject_2_int_dict['1'], spo_subject_3_int_dict['1'],
                                    spo_subject_4_int_dict['1'], spo_subject_5_int_dict['1']))
idoa_1_day_intervension_spo = IDOA(cohort, day_1_intervension_spo, min_overlap=min_overlap_val)
#idoa_1_day_intervension_spo_vector = idoa_1_day_intervension_spo.calc_idoa_vector()
idoa_1_day_intervension_spo_vector = idoa_1_day_intervension_spo.calc_idoa_vector_custom({0:13, 1:14, 2:15, 3:16})
BC_1_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_1_intervension_spo, {0:13, 1:14, 2:15, 3:16})

day_2_intervension_spo = np.vstack((spo_subject_2_int_dict['2'], spo_subject_3_int_dict['2'],
                                    spo_subject_4_int_dict['2'], spo_subject_5_int_dict['2']))
idoa_2_day_intervension_spo = IDOA(cohort, day_2_intervension_spo, min_overlap=min_overlap_val)
#idoa_2_day_intervension_spo_vector = idoa_2_day_intervension_spo.calc_idoa_vector()
idoa_2_day_intervension_spo_vector = idoa_2_day_intervension_spo.calc_idoa_vector_custom({0:13, 1:14, 2:15, 3:16})
BC_2_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_2_intervension_spo, {0:13, 1:14, 2:15, 3:16})

day_3_intervension_spo = np.vstack((spo_subject_1_int_dict['3'], spo_subject_2_int_dict['3'],
                                    spo_subject_3_int_dict['3'], spo_subject_4_int_dict['3'],
                                    spo_subject_5_int_dict['3']))
idoa_3_day_intervension_spo = IDOA(cohort, day_3_intervension_spo, min_overlap=min_overlap_val)
#idoa_3_day_intervension_spo_vector = idoa_3_day_intervension_spo.calc_idoa_vector()
idoa_3_day_intervension_spo_vector = idoa_3_day_intervension_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_3_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_3_intervension_spo,
                                                                  {0:12, 1:13, 2:14, 3:15, 4:16})

day_4_intervension_spo = np.vstack((spo_subject_1_int_dict['4'], spo_subject_2_int_dict['4'],
                                    spo_subject_3_int_dict['4'], spo_subject_4_int_dict['4'],
                                    spo_subject_5_int_dict['4']))
idoa_4_day_intervension_spo = IDOA(cohort, day_4_intervension_spo, min_overlap=min_overlap_val)
#idoa_4_day_intervension_spo_vector = idoa_4_day_intervension_spo.calc_idoa_vector()
idoa_4_day_intervension_spo_vector = idoa_4_day_intervension_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_4_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_4_intervension_spo,
                                                                  {0:12, 1:13, 2:14, 3:15, 4:16})

day_5_intervension_spo = np.vstack((spo_subject_1_int_dict['5'], spo_subject_2_int_dict['5'],
                                    spo_subject_3_int_dict['5'], spo_subject_4_int_dict['5'],
                                    spo_subject_5_int_dict['5']))
idoa_5_day_intervension_spo = IDOA(cohort, day_5_intervension_spo, min_overlap=min_overlap_val)
#idoa_5_day_intervension_spo_vector = idoa_5_day_intervension_spo.calc_idoa_vector()
idoa_5_day_intervension_spo_vector = idoa_5_day_intervension_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_5_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_5_intervension_spo,
                                                                  {0:12, 1:13, 2:14, 3:15, 4:16})

day_6_intervension_spo = np.vstack((spo_subject_2_int_dict['6'], spo_subject_3_int_dict['6'],
                                    spo_subject_4_int_dict['6'], spo_subject_5_int_dict['6']))
idoa_6_day_intervension_spo = IDOA(cohort, day_6_intervension_spo, min_overlap=min_overlap_val)
#idoa_6_day_intervension_spo_vector = idoa_6_day_intervension_spo.calc_idoa_vector()
idoa_6_day_intervension_spo_vector = idoa_6_day_intervension_spo.calc_idoa_vector_custom({0:13, 1:14, 2:15, 3:16})
BC_6_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_6_intervension_spo, {0:13, 1:14, 2:15, 3:16})

day_7_intervension_spo = np.vstack((spo_subject_1_int_dict['7'], spo_subject_2_int_dict['7'],
                                    spo_subject_3_int_dict['7'], spo_subject_4_int_dict['7'],
                                    spo_subject_5_int_dict['7']))
idoa_7_day_intervension_spo = IDOA(cohort, day_7_intervension_spo, min_overlap=min_overlap_val)
#idoa_7_day_intervension_spo_vector = idoa_7_day_intervension_spo.calc_idoa_vector()
idoa_7_day_intervension_spo_vector = idoa_7_day_intervension_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_7_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_7_intervension_spo,
                                                                  {0:12, 1:13, 2:14, 3:15, 4:16})

day_14_intervension_spo = np.vstack((spo_subject_1_int_dict['14'], spo_subject_2_int_dict['14'],
                                     spo_subject_3_int_dict['14'], spo_subject_4_int_dict['14'],
                                     spo_subject_5_int_dict['14']))
idoa_14_day_intervension_spo = IDOA(cohort, day_14_intervension_spo, min_overlap=min_overlap_val)
#idoa_14_day_intervension_spo_vector = idoa_14_day_intervension_spo.calc_idoa_vector()
idoa_14_day_intervension_spo_vector = idoa_14_day_intervension_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_14_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_14_intervension_spo,
                                                                  {0:12, 1:13, 2:14, 3:15, 4:16})

day_21_intervension_spo = np.vstack((spo_subject_1_int_dict['21'], spo_subject_2_int_dict['21'],
                                      spo_subject_3_int_dict['21'], spo_subject_4_int_dict['21'],
                                      spo_subject_5_int_dict['21']))
idoa_21_day_intervension_spo = IDOA(cohort, day_21_intervension_spo, min_overlap=min_overlap_val)
#idoa_21_day_intervension_spo_vector = idoa_21_day_intervension_spo.calc_idoa_vector()
idoa_21_day_intervension_spo_vector = idoa_21_day_intervension_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_21_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_21_intervension_spo,
                                                                   {0:12, 1:13, 2:14, 3:15, 4:16})

day_28_intervension_spo = np.vstack((spo_subject_1_int_dict['28'], spo_subject_2_int_dict['28'],
                                     spo_subject_3_int_dict['28'], spo_subject_4_int_dict['28'],
                                     spo_subject_5_int_dict['28']))
idoa_28_day_intervension_spo = IDOA(cohort, day_28_intervension_spo, min_overlap=min_overlap_val)
#idoa_28_day_intervension_spo_vector = idoa_28_day_intervension_spo.calc_idoa_vector()
idoa_28_day_intervension_spo_vector = idoa_28_day_intervension_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_28_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_28_intervension_spo,
                                                                   {0:12, 1:13, 2:14, 3:15, 4:16})

day_42_intervension_spo = np.vstack((spo_subject_1_int_dict['42'], spo_subject_2_int_dict['42'],
                                     spo_subject_3_int_dict['42'], spo_subject_4_int_dict['42'],
                                     spo_subject_5_int_dict['42']))
idoa_42_day_intervension_spo = IDOA(cohort, day_42_intervension_spo, min_overlap=min_overlap_val)
#idoa_42_day_intervension_spo_vector = idoa_42_day_intervension_spo.calc_idoa_vector()
idoa_42_day_intervension_spo_vector = idoa_42_day_intervension_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_42_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_42_intervension_spo,
                                                                  {0:12, 1:13, 2:14, 3:15, 4:16})

day_56_intervension_spo = np.vstack((spo_subject_1_int_dict['56'], spo_subject_2_int_dict['56'],
                                     spo_subject_3_int_dict['56'], spo_subject_4_int_dict['56'],
                                     spo_subject_5_int_dict['56']))
idoa_56_day_intervension_spo = IDOA(cohort, day_56_intervension_spo, min_overlap=min_overlap_val)
#idoa_56_day_intervension_spo_vector = idoa_56_day_intervension_spo.calc_idoa_vector()
idoa_56_day_intervension_spo_vector = idoa_56_day_intervension_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_56_day_intervension_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_56_intervension_spo,
                                                                   {0:12, 1:13, 2:14, 3:15, 4:16})

month_3_spo = np.vstack((spo_subject_1_month_dict['3'], spo_subject_2_month_dict['3']))
idoa_month_3_spo = IDOA(cohort, month_3_spo, min_overlap=min_overlap_val)
#idoa_month_3_spo_vector = idoa_month_3_spo.calc_idoa_vector()
idoa_month_3_spo_vector = idoa_month_3_spo.calc_idoa_vector_custom({0:12, 1:13})
BC_month_3_spo_vector = calc_bray_curtis_dissimilarity(cohort, month_3_spo, {0:12, 1:13})

month_4_spo = np.vstack((spo_subject_1_month_dict['4'], spo_subject_2_month_dict['4'],
                         spo_subject_3_month_dict['4'], spo_subject_5_month_dict['4']))
idoa_month_4_spo = IDOA(cohort, month_4_spo, min_overlap=min_overlap_val)
#idoa_month_4_spo_vector = idoa_month_4_spo.calc_idoa_vector()
idoa_month_4_spo_vector = idoa_month_4_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:16})
BC_month_4_spo_vector = calc_bray_curtis_dissimilarity(cohort, month_4_spo, {0:12, 1:13, 2:14, 3:16})

month_5_spo = np.vstack((spo_subject_1_month_dict['5'], spo_subject_2_month_dict['5'],
                         spo_subject_3_month_dict['5'], spo_subject_5_month_dict['5']))
idoa_month_5_spo = IDOA(cohort, month_5_spo, min_overlap=min_overlap_val)
#idoa_month_5_spo_vector = idoa_month_5_spo.calc_idoa_vector()
idoa_month_5_spo_vector = idoa_month_5_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:16})
BC_month_5_spo_vector = calc_bray_curtis_dissimilarity(cohort, month_5_spo, {0:12, 1:13, 2:14, 3:16})

month_6_spo = np.vstack((spo_subject_1_month_dict['6'], spo_subject_2_month_dict['6'],
                         spo_subject_3_month_dict['6'], spo_subject_5_month_dict['6']))
idoa_month_6_spo = IDOA(cohort, month_6_spo, min_overlap=min_overlap_val)
#idoa_month_6_spo_vector = idoa_month_6_spo.calc_idoa_vector()
idoa_month_6_spo_vector = idoa_month_6_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:16})
BC_month_6_spo_vector = calc_bray_curtis_dissimilarity(cohort, month_6_spo, {0:12, 1:13, 2:14, 3:16})

day_1_base_spo = np.vstack((spo_subject_1_base_dict['1'], spo_subject_2_base_dict['1'],
                            spo_subject_3_base_dict['1'], spo_subject_4_base_dict['1'],
                            spo_subject_5_base_dict['1']))
idoa_1_day_base_spo = IDOA(cohort, day_1_base_spo, min_overlap=min_overlap_val)
#idoa_1_day_base_spo_vector = idoa_1_day_base_spo.calc_idoa_vector()
idoa_1_day_base_spo_vector = idoa_1_day_base_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_1_day_base_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_1_base_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

day_2_base_spo = np.vstack((spo_subject_1_base_dict['2'], spo_subject_2_base_dict['2'],
                            spo_subject_3_base_dict['2'], spo_subject_4_base_dict['2'],
                            spo_subject_5_base_dict['2']))
idoa_2_day_base_spo = IDOA(cohort, day_2_base_spo, min_overlap=min_overlap_val)
#idoa_2_day_base_spo_vector = idoa_2_day_base_spo.calc_idoa_vector()
idoa_2_day_base_spo_vector = idoa_2_day_base_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_2_day_base_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_2_base_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

day_3_base_spo = np.vstack((spo_subject_1_base_dict['3'], spo_subject_2_base_dict['3'],
                            spo_subject_3_base_dict['3'], spo_subject_4_base_dict['3'],
                            spo_subject_5_base_dict['3']))
idoa_3_day_base_spo = IDOA(cohort, day_3_base_spo, min_overlap=min_overlap_val)
#idoa_3_day_base_spo_vector = idoa_3_day_base_spo.calc_idoa_vector()
idoa_3_day_base_spo_vector = idoa_3_day_base_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_3_day_base_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_3_base_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

day_4_base_spo = np.vstack((spo_subject_1_base_dict['4'], spo_subject_2_base_dict['4'],
                            spo_subject_3_base_dict['4'], spo_subject_4_base_dict['4'],
                            spo_subject_5_base_dict['4']))
idoa_4_day_base_spo = IDOA(cohort, day_4_base_spo, min_overlap=min_overlap_val)
#idoa_4_day_base_spo_vector = idoa_4_day_base_spo.calc_idoa_vector()
idoa_4_day_base_spo_vector = idoa_4_day_base_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_4_day_base_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_4_base_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

day_5_base_spo = np.vstack((spo_subject_2_base_dict['5'], spo_subject_3_base_dict['5'],
                            spo_subject_5_base_dict['5']))
idoa_5_day_base_spo = IDOA(cohort, day_5_base_spo, min_overlap=min_overlap_val)
#idoa_5_day_base_spo_vector = idoa_5_day_base_spo.calc_idoa_vector()
idoa_5_day_base_spo_vector = idoa_5_day_base_spo.calc_idoa_vector_custom({0:13, 1:14, 2:15})
BC_5_day_base_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_5_base_spo, {0:13, 1:14, 2:15})

day_6_base_spo = np.vstack((spo_subject_2_base_dict['6'], spo_subject_3_base_dict['6'],
                            spo_subject_4_base_dict['6'], spo_subject_5_base_dict['6']))
idoa_6_day_base_spo = IDOA(cohort, day_6_base_spo, min_overlap=min_overlap_val)
#idoa_6_day_base_spo_vector = idoa_6_day_base_spo.calc_idoa_vector()
idoa_6_day_base_spo_vector = idoa_6_day_base_spo.calc_idoa_vector_custom({0:13, 1:14, 2:15, 3:16})
BC_6_day_base_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_6_base_spo, {0:13, 1:14, 2:15, 3:16})

day_7_base_spo = np.vstack((spo_subject_1_base_dict['7'], spo_subject_2_base_dict['7'],
                            spo_subject_3_base_dict['7'], spo_subject_4_base_dict['7'],
                            spo_subject_5_base_dict['7']))
idoa_7_day_base_spo = IDOA(cohort, day_7_base_spo, min_overlap=min_overlap_val)
#idoa_7_day_base_spo_vector = idoa_7_day_base_spo.calc_idoa_vector()
idoa_7_day_base_spo_vector = idoa_7_day_base_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_7_day_base_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_7_base_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

day_1_ant_spo = np.vstack((spo_subject_2_ant_dict['1'], spo_subject_3_ant_dict['1'],
                           spo_subject_4_ant_dict['1'], spo_subject_5_ant_dict['1']))
idoa_1_day_ant_spo = IDOA(cohort, day_1_ant_spo, min_overlap=min_overlap_val)
#idoa_1_day_ant_spo_vector = idoa_1_day_ant_spo.calc_idoa_vector()
idoa_1_day_ant_spo_vector = idoa_1_day_ant_spo.calc_idoa_vector_custom({0:13, 1:14, 2:15, 3:16})
BC_1_day_ant_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_1_ant_spo, {0:13, 1:14, 2:15, 3:16})

day_2_ant_spo = np.vstack((spo_subject_1_ant_dict['2'], spo_subject_2_ant_dict['2'],
                           spo_subject_3_ant_dict['2'], spo_subject_4_ant_dict['2'],
                           spo_subject_5_ant_dict['2']))
idoa_2_day_ant_spo = IDOA(cohort, day_2_ant_spo, min_overlap=min_overlap_val)
#idoa_2_day_ant_spo_vector = idoa_2_day_ant_spo.calc_idoa_vector()
idoa_2_day_ant_spo_vector = idoa_2_day_ant_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_2_day_ant_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_2_ant_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

day_3_ant_spo = np.vstack((spo_subject_1_ant_dict['3'], spo_subject_2_ant_dict['3'],
                           spo_subject_3_ant_dict['3'], spo_subject_4_ant_dict['3'],
                           spo_subject_5_ant_dict['3']))
idoa_3_day_ant_spo = IDOA(cohort, day_3_ant_spo, min_overlap=min_overlap_val)
#idoa_3_day_ant_spo_vector = idoa_3_day_ant_spo.calc_idoa_vector()
idoa_3_day_ant_spo_vector = idoa_3_day_ant_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_3_day_ant_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_3_ant_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

day_4_ant_spo = np.vstack((spo_subject_1_ant_dict['4'], spo_subject_2_ant_dict['4'],
                           spo_subject_3_ant_dict['4'], spo_subject_4_ant_dict['4'],
                           spo_subject_5_ant_dict['4']))
idoa_4_day_ant_spo = IDOA(cohort, day_4_ant_spo, min_overlap=min_overlap_val)
#idoa_4_day_ant_spo_vector = idoa_4_day_ant_spo.calc_idoa_vector()
idoa_4_day_ant_spo_vector = idoa_4_day_ant_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_4_day_ant_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_4_ant_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

day_5_ant_spo = np.vstack((spo_subject_1_ant_dict['5'], spo_subject_2_ant_dict['5'],
                           spo_subject_3_ant_dict['5'], spo_subject_4_ant_dict['5'],
                           spo_subject_5_ant_dict['5']))
idoa_5_day_ant_spo = IDOA(cohort, day_5_ant_spo, min_overlap=min_overlap_val, zero_overlap=0)
#idoa_5_day_ant_spo_vector = idoa_5_day_ant_spo.calc_idoa_vector()
idoa_5_day_ant_spo_vector = idoa_5_day_ant_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})

BC_5_day_ant_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_5_ant_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

day_6_ant_spo = np.vstack((spo_subject_1_ant_dict['6'], spo_subject_2_ant_dict['6'],
                           spo_subject_3_ant_dict['6'], spo_subject_4_ant_dict['6'],
                           spo_subject_5_ant_dict['6']))
idoa_6_day_ant_spo = IDOA(cohort, day_6_ant_spo, min_overlap=min_overlap_val)
#idoa_6_day_ant_spo_vector = idoa_6_day_ant_spo.calc_idoa_vector()
idoa_6_day_ant_spo_vector = idoa_6_day_ant_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_6_day_ant_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_6_ant_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

day_7_ant_spo = np.vstack((spo_subject_1_ant_dict['7'], spo_subject_2_ant_dict['7'],
                           spo_subject_3_ant_dict['7'], spo_subject_4_ant_dict['7'],
                           spo_subject_5_ant_dict['7']))
idoa_7_day_ant_spo = IDOA(cohort, day_7_ant_spo, min_overlap=min_overlap_val)
#idoa_7_day_ant_spo_vector = idoa_7_day_ant_spo.calc_idoa_vector()
idoa_7_day_ant_spo_vector = idoa_7_day_ant_spo.calc_idoa_vector_custom({0:12, 1:13, 2:14, 3:15, 4:16})
BC_7_day_ant_spo_vector = calc_bray_curtis_dissimilarity(cohort, day_7_ant_spo, {0:12, 1:13, 2:14, 3:15, 4:16})

import numpy as np
import matplotlib.pyplot as plt

# Create a list of vectors
aFMT_idoa_int = [idoa_1_day_intervension_aFMT_vector, idoa_2_day_intervension_aFMT_vector,
                 idoa_3_day_intervension_aFMT_vector, idoa_4_day_intervension_aFMT_vector,
                 idoa_5_day_intervension_aFMT_vector, idoa_6_day_intervension_aFMT_vector,
                 idoa_7_day_intervension_aFMT_vector, idoa_14_day_intervension_aFMT_vector,
                 idoa_21_day_intervension_aFMT_vector, idoa_28_day_intervension_aFMT_vector,
                 idoa_42_day_intervension_aFMT_vector, idoa_56_day_intervension_aFMT_vector]

aFMT_idoa_base = [idoa_1_day_base_aFMT_vector, idoa_2_day_base_aFMT_vector,
                  idoa_3_day_base_aFMT_vector, idoa_4_day_base_aFMT_vector,
                  idoa_5_day_base_aFMT_vector, idoa_6_day_base_aFMT_vector,
                  idoa_7_day_base_aFMT_vector]

aFMT_idoa_ant = [idoa_1_day_ant_aFMT_vector, idoa_2_day_ant_aFMT_vector,
                 idoa_3_day_ant_aFMT_vector, idoa_4_day_ant_aFMT_vector,
                 idoa_5_day_ant_aFMT_vector, idoa_6_day_ant_aFMT_vector,
                 idoa_7_day_ant_aFMT_vector]

aFMT_idoa_month = [idoa_month_1_aFMT_vector, idoa_month_2_aFMT_vector, idoa_month_3_aFMT_vector,
                   idoa_month_4_aFMT_vector, idoa_month_5_aFMT_vector, idoa_month_6_aFMT_vector]

pro_idoa_int = [idoa_1_day_intervension_pro_vector, idoa_2_day_intervension_pro_vector, idoa_3_day_intervension_pro_vector,
                idoa_4_day_intervension_pro_vector, idoa_5_day_intervension_pro_vector, idoa_6_day_intervension_pro_vector,
                idoa_7_day_intervension_pro_vector, idoa_14_day_intervension_pro_vector, idoa_21_day_intervension_pro_vector,
                idoa_28_day_intervension_pro_vector, idoa_42_day_intervension_pro_vector,
                idoa_56_day_intervension_pro_vector]

pro_idoa_base = [idoa_1_day_base_pro_vector, idoa_2_day_base_pro_vector,
                 idoa_3_day_base_pro_vector, idoa_4_day_base_pro_vector,
                 idoa_5_day_base_pro_vector, idoa_6_day_base_pro_vector,
                 idoa_7_day_base_pro_vector]

pro_idoa_ant = [idoa_1_day_ant_pro_vector, idoa_2_day_ant_pro_vector,
                idoa_3_day_ant_pro_vector, idoa_4_day_ant_pro_vector,
                idoa_5_day_ant_pro_vector, idoa_6_day_ant_pro_vector,
                idoa_7_day_ant_pro_vector]

pro_idoa_month = [idoa_month_2_pro_vector, idoa_month_3_pro_vector, idoa_month_4_pro_vector,
                  idoa_month_5_pro_vector, idoa_month_6_pro_vector]

spo_idoa_int = [idoa_1_day_intervension_spo_vector, idoa_2_day_intervension_spo_vector, idoa_3_day_intervension_spo_vector,
                idoa_4_day_intervension_spo_vector, idoa_5_day_intervension_spo_vector, idoa_6_day_intervension_spo_vector,
                idoa_7_day_intervension_spo_vector, idoa_14_day_intervension_spo_vector, idoa_21_day_intervension_spo_vector,
                idoa_28_day_intervension_spo_vector, idoa_42_day_intervension_spo_vector,
                idoa_56_day_intervension_spo_vector]

spo_idoa_base = [idoa_1_day_base_spo_vector, idoa_2_day_base_spo_vector,
                 idoa_3_day_base_spo_vector, idoa_4_day_base_spo_vector,
                 idoa_5_day_base_spo_vector, idoa_6_day_base_spo_vector,
                 idoa_7_day_base_spo_vector]

spo_idoa_ant = [idoa_1_day_ant_spo_vector, idoa_2_day_ant_spo_vector,
                idoa_3_day_ant_spo_vector, idoa_4_day_ant_spo_vector,
                idoa_5_day_ant_spo_vector, idoa_6_day_ant_spo_vector,
                idoa_7_day_ant_spo_vector]

spo_idoa_month = [idoa_month_3_spo_vector, idoa_month_4_spo_vector,
                  idoa_month_5_spo_vector, idoa_month_6_spo_vector]

# Calculate the mean and standard deviation of each vector
means_aFMT_int = [np.mean(vector) for vector in aFMT_idoa_int]
stds_aFMT_int = [np.std(vector)/ np.sqrt(len(vector)) for vector in aFMT_idoa_int]
means_aFMT_base = [np.mean(vector) for vector in aFMT_idoa_base]
stds_aFMT_base = [np.std(vector)/ np.sqrt(len(vector)) for vector in aFMT_idoa_base]
means_aFMT_ant = [np.mean(vector) for vector in aFMT_idoa_ant]
stds_aFMT_ant = [np.std(vector)/ np.sqrt(len(vector)) for vector in aFMT_idoa_ant]
means_aFMT_month = [np.mean(vector) for vector in aFMT_idoa_month]
stds_aFMT_month = [np.std(vector)/ np.sqrt(len(vector)) for vector in aFMT_idoa_month]

means_pro_int = [np.mean(vector) for vector in pro_idoa_int]
stds_pro_int = [np.std(vector)/ np.sqrt(len(vector)) for vector in pro_idoa_int]
means_pro_base = [np.mean(vector) for vector in pro_idoa_base]
stds_pro_base = [np.std(vector)/ np.sqrt(len(vector)) for vector in pro_idoa_base]
means_pro_ant = [np.mean(vector) for vector in pro_idoa_ant]
stds_pro_ant = [np.std(vector)/ np.sqrt(len(vector)) for vector in pro_idoa_ant]
means_pro_month = [np.mean(vector) for vector in pro_idoa_month]
stds_pro_month = [np.std(vector)/ np.sqrt(len(vector)) for vector in pro_idoa_month]

means_spo_int = [np.mean(vector) for vector in spo_idoa_int]
stds_spo_int = [np.std(vector)/ np.sqrt(len(vector)) for vector in spo_idoa_int]
means_spo_base = [np.mean(vector) for vector in spo_idoa_base]
stds_spo_base = [np.std(vector)/ np.sqrt(len(vector)) for vector in spo_idoa_base]
means_spo_ant = [np.mean(vector) for vector in spo_idoa_ant]
stds_spo_ant = [np.std(vector)/ np.sqrt(len(vector)) for vector in spo_idoa_ant]
means_spo_month = [np.mean(vector) for vector in spo_idoa_month]
stds_spo_month = [np.std(vector)/ np.sqrt(len(vector)) for vector in spo_idoa_month]


# Create a list of the number of vectors
days_int = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 42, 56]
days_base = [-13, -12, -11, -10, -9, -8, -7]
days_ant = [-6, -5, -4, -3, -2, -1, 0]
month = [86, 116, 146, 176, 206, 236]
month_no_1 = [116, 146, 176, 206, 236]
month_no_2 = [146, 176, 206, 236]

# Set the figure size
fig, ax = plt.subplots(1, 2, figsize=(18,7), sharey=True, gridspec_kw={'wspace': 0.1, 'hspace': 0.8})

# Plot the mean as a scatter plot with error bars for the standard deviation
ax[0].errorbar(days_int, means_aFMT_int, yerr=stds_aFMT_int, fmt='o', capsize=5, ls='none', color='brown')
ax[0].plot(days_int, means_aFMT_int, '-o', color='brown')
ax[0].errorbar(days_base, means_aFMT_base, yerr=stds_aFMT_base, fmt='o', capsize=5, ls='none', color='brown')
ax[0].plot(days_base, means_aFMT_base, '-o', color='brown')
ax[0].errorbar(days_ant, means_aFMT_ant, yerr=stds_aFMT_ant, fmt='o', capsize=5, ls='none', color='brown')
ax[0].plot(days_ant, means_aFMT_ant, '-o', color='brown')

ax[0].errorbar(days_int, means_pro_int, yerr=stds_pro_int, fmt='o', capsize=5, ls='none', color='green')
ax[0].plot(days_int, means_pro_int, '-o', color='green')
ax[0].errorbar(days_base, means_pro_base, yerr=stds_pro_base, fmt='o', capsize=5, ls='none', color='green')
ax[0].plot(days_base, means_pro_base, '-o', color='green')
ax[0].errorbar(days_ant, means_pro_ant, yerr=stds_pro_ant, fmt='o', capsize=5, ls='none', color='green')
ax[0].plot(days_ant, means_pro_ant, '-o', color='green')

ax[0].errorbar(days_int, means_spo_int, yerr=stds_spo_int, fmt='o', capsize=5, ls='none', color='chocolate')
ax[0].plot(days_int, means_spo_int, '-o', color='chocolate')
ax[0].errorbar(days_base, means_spo_base, yerr=stds_spo_base, fmt='o', capsize=5, ls='none', color='chocolate')
ax[0].plot(days_base, means_spo_base, '-o', color='chocolate')
ax[0].errorbar(days_ant, means_spo_ant, yerr=stds_spo_ant, fmt='o', capsize=5, ls='none', color='chocolate')
ax[0].plot(days_ant, means_spo_ant, '-o', color='chocolate')

ax[0].set_ylim(-0.5, 0.1)
y_min, y_max = ax[0].get_ylim()
ax[0].set_ylabel('Mean IDOA', fontsize='20')
ax[0].spines['right'].set_color('white')
ax[0].axvline(x=-7, color='k', linestyle='--')
ax[0].axvline(x=0, color='k', linestyle='--')
ax[0].fill_betweenx([y_min,y_max], -7, 0, color='gray', alpha=0.2)

ax[1].errorbar(month[2:], means_aFMT_month[2:], yerr=stds_aFMT_month[2:], fmt='o', capsize=5, ls='none', color='brown')
ax[1].plot(month[2:], means_aFMT_month[2:], '-o', color='brown', label='aFMT')

ax[1].errorbar(month_no_1, means_pro_month, yerr=stds_pro_month, fmt='o', capsize=5, ls='none', color='green')
ax[1].plot(month_no_1, means_pro_month, '-o', color='green', label='Probiotics')

ax[1].errorbar(month_no_2, means_spo_month, yerr=stds_spo_month, fmt='o', capsize=5, ls='none', color='chocolate')
ax[1].plot(month_no_2, means_spo_month, '-o', color='chocolate', label='Spontaneous')

ax[1].set_ylim(-0.5, 0.1)
ax[1].get_yaxis().set_visible(False)
ax[1].spines['left'].set_color('white')
ax[1].legend(loc='lower right', fontsize='15')

# Set the x-axis label for both subplots
fig.text(0.5, 0.04, 'Days', ha='center', fontsize='20')

# Add some padding between the plots and the x-axis label
fig.subplots_adjust(bottom=0.15)
plt.show()

aFMT_idoa_ant = [idoa_1_day_ant_aFMT.dissimilarity_overlap_container,
                 idoa_2_day_ant_aFMT.dissimilarity_overlap_container,
                 idoa_3_day_ant_aFMT.dissimilarity_overlap_container,
                 idoa_4_day_ant_aFMT.dissimilarity_overlap_container,
                 idoa_5_day_ant_aFMT.dissimilarity_overlap_container,
                 idoa_6_day_ant_aFMT.dissimilarity_overlap_container,
                 idoa_7_day_ant_aFMT.dissimilarity_overlap_container]

aFMT_idoa_int = [idoa_1_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_2_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_3_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_4_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_5_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_6_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_7_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_14_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_21_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_28_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_42_day_intervension_aFMT.dissimilarity_overlap_container,
                 idoa_56_day_intervension_aFMT.dissimilarity_overlap_container]

aFMT_idoa_month = [idoa_month_1_aFMT.dissimilarity_overlap_container,
                   idoa_month_2_aFMT.dissimilarity_overlap_container,
                   idoa_month_3_aFMT.dissimilarity_overlap_container,
                   idoa_month_4_aFMT.dissimilarity_overlap_container,
                   idoa_month_5_aFMT.dissimilarity_overlap_container,
                   idoa_month_6_aFMT.dissimilarity_overlap_container]

fig, ax = plt.subplots(7, 6, figsize=(10, 5))
# Flatten the axs array to access each subplot individually
ax = ax.ravel()


def add_plot(ax_ind, idoa_ind, cont_ind, container):
    slope, intercept = np.polyfit(container[idoa_ind][cont_ind][0, :], container[idoa_ind][cont_ind][1, :], 1)
    x = np.array([container[idoa_ind][cont_ind][0, :].min(), container[idoa_ind][cont_ind][0, :].max()])
    y = slope * x + intercept
    ax[ax_ind].plot(x, y, color='red')
    ax[ax_ind].scatter(container[idoa_ind][cont_ind][0, :], container[idoa_ind][cont_ind][1, :])


add_plot(2, 0, 0, aFMT_idoa_ant)
add_plot(3, 0, 1, aFMT_idoa_ant)

add_plot(6, 1, 0, aFMT_idoa_ant)
add_plot(7, 1, 1, aFMT_idoa_ant)
add_plot(8, 1, 2, aFMT_idoa_ant)
add_plot(9, 1, 3, aFMT_idoa_ant)
add_plot(10, 1, 4, aFMT_idoa_ant)
add_plot(11, 1, 5, aFMT_idoa_ant)

add_plot(14, 2, 0, aFMT_idoa_ant)
add_plot(16, 2, 1, aFMT_idoa_ant)
add_plot(17, 2, 2, aFMT_idoa_ant)

add_plot(19, 3, 0, aFMT_idoa_ant)
add_plot(20, 3, 1, aFMT_idoa_ant)
add_plot(22, 3, 2, aFMT_idoa_ant)
add_plot(23, 3, 3, aFMT_idoa_ant)

add_plot(25, 4, 0, aFMT_idoa_ant)
add_plot(26, 4, 1, aFMT_idoa_ant)
add_plot(27, 4, 2, aFMT_idoa_ant)
add_plot(28, 4, 3, aFMT_idoa_ant)
add_plot(29, 4, 4, aFMT_idoa_ant)

add_plot(31, 5, 0, aFMT_idoa_ant)
add_plot(32, 5, 1, aFMT_idoa_ant)
add_plot(33, 5, 2, aFMT_idoa_ant)
add_plot(34, 5, 3, aFMT_idoa_ant)
add_plot(35, 5, 4, aFMT_idoa_ant)

add_plot(37, 6, 0, aFMT_idoa_ant)
add_plot(38, 6, 1, aFMT_idoa_ant)
add_plot(39, 6, 2, aFMT_idoa_ant)
add_plot(41, 6, 3, aFMT_idoa_ant)

for x in ax.flat:
    x.set_xticklabels([])
    x.set_yticklabels([])

for x in ax.flatten():
    x.tick_params(bottom=False, left=False)

fig.text(0.5, 0.004, 'Subjects', ha='center', va='center', fontsize=15)
fig.text(0.06, 0.5, 'Days', ha='center', va='center', rotation='vertical', fontsize=15)

for i, label in enumerate(['1', '2', '3', '4', '5', '6']):
    fig.text(0.18 + i * 0.134, 0.04, label, ha='center', va='center')

for i, label in enumerate(['0', '-1', '-2', '-3', '-4', '-5', '-6']):
    fig.text(0.1, 0.16 + i * 0.11, label, ha='center', va='center', rotation='vertical')

fig, ax = plt.subplots(12, 6, figsize=(10, 10))
# Flatten the axs array to access each subplot individually
ax = ax.ravel()

add_plot(0, 0, 0, aFMT_idoa_int)
add_plot(1, 0, 1, aFMT_idoa_int)
add_plot(2, 0, 2, aFMT_idoa_int)
add_plot(4, 0, 3, aFMT_idoa_int)
add_plot(5, 0, 4, aFMT_idoa_int)

add_plot(6, 1, 0, aFMT_idoa_int)
add_plot(7, 1, 1, aFMT_idoa_int)
add_plot(8, 1, 2, aFMT_idoa_int)
add_plot(10, 1, 3, aFMT_idoa_int)
add_plot(11, 1, 4, aFMT_idoa_int)

add_plot(12, 2, 0, aFMT_idoa_int)
add_plot(13, 2, 1, aFMT_idoa_int)
add_plot(14, 2, 2, aFMT_idoa_int)
add_plot(15, 2, 3, aFMT_idoa_int)
add_plot(16, 2, 4, aFMT_idoa_int)
add_plot(17, 2, 5, aFMT_idoa_int)

add_plot(18, 3, 0, aFMT_idoa_int)
add_plot(19, 3, 1, aFMT_idoa_int)
add_plot(20, 3, 2, aFMT_idoa_int)
add_plot(21, 3, 3, aFMT_idoa_int)
add_plot(22, 3, 4, aFMT_idoa_int)
add_plot(23, 3, 5, aFMT_idoa_int)

add_plot(24, 4, 0, aFMT_idoa_int)
add_plot(25, 4, 1, aFMT_idoa_int)
add_plot(26, 4, 2, aFMT_idoa_int)
add_plot(27, 4, 3, aFMT_idoa_int)
add_plot(28, 4, 4, aFMT_idoa_int)
add_plot(29, 4, 5, aFMT_idoa_int)

add_plot(31, 5, 0, aFMT_idoa_int)
add_plot(32, 5, 1, aFMT_idoa_int)
add_plot(33, 5, 2, aFMT_idoa_int)
add_plot(34, 5, 3, aFMT_idoa_int)

add_plot(37, 6, 0, aFMT_idoa_int)
add_plot(38, 6, 1, aFMT_idoa_int)
add_plot(39, 6, 2, aFMT_idoa_int)
add_plot(40, 6, 3, aFMT_idoa_int)

add_plot(42, 7, 0, aFMT_idoa_int)
add_plot(43, 7, 1, aFMT_idoa_int)
add_plot(44, 7, 2, aFMT_idoa_int)
add_plot(45, 7, 3, aFMT_idoa_int)
add_plot(46, 7, 4, aFMT_idoa_int)
add_plot(47, 7, 5, aFMT_idoa_int)

add_plot(48, 8, 0, aFMT_idoa_int)
add_plot(49, 8, 1, aFMT_idoa_int)
add_plot(50, 8, 2, aFMT_idoa_int)
add_plot(51, 8, 3, aFMT_idoa_int)
add_plot(52, 8, 4, aFMT_idoa_int)
add_plot(53, 8, 5, aFMT_idoa_int)

add_plot(54, 9, 0, aFMT_idoa_int)
add_plot(55, 9, 1, aFMT_idoa_int)
add_plot(56, 9, 2, aFMT_idoa_int)
add_plot(57, 9, 3, aFMT_idoa_int)
add_plot(58, 9, 4, aFMT_idoa_int)
add_plot(59, 9, 5, aFMT_idoa_int)

add_plot(60, 10, 0, aFMT_idoa_int)
add_plot(61, 10, 1, aFMT_idoa_int)
add_plot(62, 10, 2, aFMT_idoa_int)
add_plot(63, 10, 3, aFMT_idoa_int)
add_plot(64, 10, 4, aFMT_idoa_int)
add_plot(65, 10, 5, aFMT_idoa_int)

add_plot(66, 11, 0, aFMT_idoa_int)
add_plot(67, 11, 1, aFMT_idoa_int)
add_plot(68, 11, 2, aFMT_idoa_int)
add_plot(69, 11, 3, aFMT_idoa_int)
add_plot(70, 11, 4, aFMT_idoa_int)
add_plot(71, 11, 5, aFMT_idoa_int)

for x in ax.flat:
    x.set_xticklabels([])
    x.set_yticklabels([])

for x in ax.flatten():
    x.tick_params(bottom=False, left=False)

fig.text(0.5, 0.04, 'Subjects', ha='center', va='center', fontsize=15)
fig.text(0.06, 0.5, 'Days', ha='center', va='center', rotation='vertical', fontsize=15)

for i, label in enumerate(['1', '2', '3', '4', '5', '6']):
    fig.text(0.18 + i * 0.134, 0.1, label, ha='center', va='center')

for i, label in enumerate(['56', '42', '28', '21', '14', '7', '6', '5', '4', '3', '2', '1']):
    fig.text(0.1, 0.16 + i * 0.063, label, ha='center', va='center', rotation='vertical')

fig, ax = plt.subplots(6, 6, figsize=(10, 6))
# Flatten the axs array to access each subplot individually
ax = ax.ravel()

# add_plot(0, 0, 0, aFMT_idoa_month)

add_plot(6, 1, 0, aFMT_idoa_month)
add_plot(7, 1, 1, aFMT_idoa_month)
add_plot(8, 1, 2, aFMT_idoa_month)

add_plot(12, 2, 0, aFMT_idoa_month)
add_plot(13, 2, 1, aFMT_idoa_month)
add_plot(14, 2, 2, aFMT_idoa_month)
add_plot(15, 2, 3, aFMT_idoa_month)

add_plot(18, 3, 0, aFMT_idoa_month)
add_plot(19, 3, 1, aFMT_idoa_month)
add_plot(20, 3, 2, aFMT_idoa_month)
add_plot(21, 3, 3, aFMT_idoa_month)
add_plot(22, 3, 4, aFMT_idoa_month)

add_plot(25, 4, 0, aFMT_idoa_month)
add_plot(26, 4, 1, aFMT_idoa_month)
add_plot(27, 4, 2, aFMT_idoa_month)
add_plot(28, 4, 3, aFMT_idoa_month)

add_plot(30, 5, 0, aFMT_idoa_month)
add_plot(31, 5, 1, aFMT_idoa_month)
add_plot(33, 5, 2, aFMT_idoa_month)
add_plot(34, 5, 3, aFMT_idoa_month)

for x in ax.flat:
    x.set_xticklabels([])
    x.set_yticklabels([])

for x in ax.flatten():
    x.tick_params(bottom=False, left=False)

fig.text(0.5, 0.04, 'Subjects', ha='center', va='center', fontsize=15)
fig.text(0.06, 0.5, 'Days', ha='center', va='center', rotation='vertical', fontsize=15)

for i, label in enumerate(['1', '2', '3', '4', '5', '6']):
    fig.text(0.18 + i * 0.134, 0.1, label, ha='center', va='center')

for i, label in enumerate(['236', '206', '176', '146', '116', '86']):
    fig.text(0.1, 0.18 + i * 0.125, label, ha='center', va='center', rotation='vertical')

plt.show()

pro_idoa_ant = [idoa_1_day_ant_pro.dissimilarity_overlap_container,
                idoa_2_day_ant_pro.dissimilarity_overlap_container,
                idoa_3_day_ant_pro.dissimilarity_overlap_container,
                idoa_4_day_ant_pro.dissimilarity_overlap_container,
                idoa_5_day_ant_pro.dissimilarity_overlap_container,
                idoa_6_day_ant_pro.dissimilarity_overlap_container,
                idoa_7_day_ant_pro.dissimilarity_overlap_container]

pro_idoa_int = [idoa_1_day_intervension_pro.dissimilarity_overlap_container,
                idoa_2_day_intervension_pro.dissimilarity_overlap_container,
                idoa_3_day_intervension_pro.dissimilarity_overlap_container,
                idoa_4_day_intervension_pro.dissimilarity_overlap_container,
                idoa_5_day_intervension_pro.dissimilarity_overlap_container,
                idoa_6_day_intervension_pro.dissimilarity_overlap_container,
                idoa_7_day_intervension_pro.dissimilarity_overlap_container,
                idoa_14_day_intervension_pro.dissimilarity_overlap_container,
                idoa_21_day_intervension_pro.dissimilarity_overlap_container,
                idoa_28_day_intervension_pro.dissimilarity_overlap_container,
                idoa_42_day_intervension_pro.dissimilarity_overlap_container,
                idoa_56_day_intervension_pro.dissimilarity_overlap_container]

pro_idoa_month = [idoa_month_2_pro.dissimilarity_overlap_container,
                  idoa_month_3_pro.dissimilarity_overlap_container,
                  idoa_month_4_pro.dissimilarity_overlap_container,
                  idoa_month_5_pro.dissimilarity_overlap_container,
                  idoa_month_6_pro.dissimilarity_overlap_container]

fig, ax = plt.subplots(7, 6, figsize=(10, 5))
# Flatten the axs array to access each subplot individually
ax = ax.ravel()


def add_plot(ax_ind, idoa_ind, cont_ind, container):
    slope, intercept = np.polyfit(container[idoa_ind][cont_ind][0, :], container[idoa_ind][cont_ind][1, :], 1)
    x = np.array([container[idoa_ind][cont_ind][0, :].min(), container[idoa_ind][cont_ind][0, :].max()])
    y = slope * x + intercept
    ax[ax_ind].plot(x, y, color='red')
    ax[ax_ind].scatter(container[idoa_ind][cont_ind][0, :], container[idoa_ind][cont_ind][1, :])


add_plot(0, 0, 0, pro_idoa_ant)
add_plot(2, 0, 1, pro_idoa_ant)
add_plot(3, 0, 2, pro_idoa_ant)
add_plot(4, 0, 3, pro_idoa_ant)
add_plot(5, 0, 4, pro_idoa_ant)

add_plot(6, 1, 0, pro_idoa_ant)
add_plot(7, 1, 1, pro_idoa_ant)
add_plot(8, 1, 2, pro_idoa_ant)
add_plot(9, 1, 3, pro_idoa_ant)
add_plot(10, 1, 4, pro_idoa_ant)
add_plot(11, 1, 5, pro_idoa_ant)

add_plot(12, 2, 0, pro_idoa_ant)
add_plot(13, 2, 1, pro_idoa_ant)
add_plot(14, 2, 2, pro_idoa_ant)
add_plot(15, 2, 3, pro_idoa_ant)
add_plot(17, 2, 4, pro_idoa_ant)

add_plot(18, 3, 0, pro_idoa_ant)
add_plot(19, 3, 1, pro_idoa_ant)
add_plot(20, 3, 2, pro_idoa_ant)
add_plot(21, 3, 3, pro_idoa_ant)
add_plot(23, 3, 4, pro_idoa_ant)

add_plot(24, 4, 0, pro_idoa_ant)
add_plot(25, 4, 1, pro_idoa_ant)
add_plot(27, 4, 2, pro_idoa_ant)
add_plot(29, 4, 3, pro_idoa_ant)

add_plot(30, 5, 0, pro_idoa_ant)
add_plot(31, 5, 1, pro_idoa_ant)
add_plot(32, 5, 2, pro_idoa_ant)
add_plot(33, 5, 3, pro_idoa_ant)
add_plot(35, 5, 4, pro_idoa_ant)

add_plot(36, 6, 0, pro_idoa_ant)
add_plot(37, 6, 1, pro_idoa_ant)
add_plot(39, 6, 2, pro_idoa_ant)
add_plot(41, 6, 3, pro_idoa_ant)

for x in ax.flat:
    x.set_xticklabels([])
    x.set_yticklabels([])

for x in ax.flatten():
    x.tick_params(bottom=False, left=False)

fig.text(0.5, 0.004, 'Subjects', ha='center', va='center', fontsize=15)
fig.text(0.06, 0.5, 'Days', ha='center', va='center', rotation='vertical', fontsize=15)

for i, label in enumerate(['1', '2', '3', '4', '5', '7']):
    fig.text(0.18 + i * 0.134, 0.04, label, ha='center', va='center')

for i, label in enumerate(['0', '-1', '-2', '-3', '-4', '-5', '-6']):
    fig.text(0.1, 0.16 + i * 0.11, label, ha='center', va='center', rotation='vertical')

fig, ax = plt.subplots(12, 6, figsize=(10, 10))
# Flatten the axs array to access each subplot individually
ax = ax.ravel()

add_plot(0, 0, 0, pro_idoa_int)
add_plot(1, 0, 1, pro_idoa_int)
add_plot(2, 0, 2, pro_idoa_int)
add_plot(4, 0, 3, pro_idoa_int)
add_plot(5, 0, 4, pro_idoa_int)

add_plot(6, 1, 0, pro_idoa_int)
add_plot(7, 1, 1, pro_idoa_int)
add_plot(8, 1, 2, pro_idoa_int)
add_plot(10, 1, 3, pro_idoa_int)
add_plot(11, 1, 4, pro_idoa_int)

add_plot(12, 2, 0, pro_idoa_int)
add_plot(14, 2, 1, pro_idoa_int)
add_plot(16, 2, 2, pro_idoa_int)
add_plot(17, 2, 3, pro_idoa_int)

add_plot(18, 3, 0, pro_idoa_int)
add_plot(19, 3, 1, pro_idoa_int)
add_plot(20, 3, 2, pro_idoa_int)
add_plot(21, 3, 3, pro_idoa_int)
add_plot(22, 3, 4, pro_idoa_int)
add_plot(23, 3, 5, pro_idoa_int)

add_plot(24, 4, 0, pro_idoa_int)
add_plot(25, 4, 1, pro_idoa_int)
add_plot(26, 4, 2, pro_idoa_int)
add_plot(27, 4, 3, pro_idoa_int)
add_plot(28, 4, 4, pro_idoa_int)
add_plot(29, 4, 5, pro_idoa_int)

add_plot(30, 5, 0, pro_idoa_int)
add_plot(31, 5, 1, pro_idoa_int)
add_plot(32, 5, 2, pro_idoa_int)
add_plot(33, 5, 3, pro_idoa_int)
add_plot(34, 5, 4, pro_idoa_int)
add_plot(35, 5, 5, pro_idoa_int)

add_plot(36, 6, 0, pro_idoa_int)
add_plot(40, 6, 1, pro_idoa_int)
add_plot(41, 6, 2, pro_idoa_int)

add_plot(42, 7, 0, pro_idoa_int)
add_plot(43, 7, 1, pro_idoa_int)
add_plot(44, 7, 2, pro_idoa_int)
add_plot(45, 7, 3, pro_idoa_int)
add_plot(46, 7, 4, pro_idoa_int)
add_plot(47, 7, 5, pro_idoa_int)

add_plot(48, 8, 0, pro_idoa_int)
add_plot(49, 8, 1, pro_idoa_int)
add_plot(50, 8, 2, pro_idoa_int)
add_plot(51, 8, 3, pro_idoa_int)
add_plot(52, 8, 4, pro_idoa_int)
add_plot(53, 8, 5, pro_idoa_int)

add_plot(54, 9, 0, pro_idoa_int)
add_plot(56, 9, 1, pro_idoa_int)
add_plot(57, 9, 2, pro_idoa_int)
add_plot(58, 9, 3, pro_idoa_int)
add_plot(59, 9, 4, pro_idoa_int)

add_plot(60, 10, 0, pro_idoa_int)
add_plot(62, 10, 1, pro_idoa_int)
add_plot(63, 10, 2, pro_idoa_int)
add_plot(64, 10, 3, pro_idoa_int)
add_plot(65, 10, 4, pro_idoa_int)

add_plot(66, 11, 0, pro_idoa_int)
add_plot(68, 11, 1, pro_idoa_int)
add_plot(69, 11, 2, pro_idoa_int)
add_plot(70, 11, 3, pro_idoa_int)
add_plot(71, 11, 4, pro_idoa_int)

for x in ax.flat:
    x.set_xticklabels([])
    x.set_yticklabels([])

for x in ax.flatten():
    x.tick_params(bottom=False, left=False)

fig.text(0.5, 0.04, 'Subjects', ha='center', va='center', fontsize=15)
fig.text(0.06, 0.5, 'Days', ha='center', va='center', rotation='vertical', fontsize=15)

for i, label in enumerate(['1', '2', '3', '4', '5', '7']):
    fig.text(0.18 + i * 0.134, 0.1, label, ha='center', va='center')

for i, label in enumerate(['56', '42', '28', '21', '14', '7', '6', '5', '4', '3', '2', '1']):
    fig.text(0.1, 0.16 + i * 0.063, label, ha='center', va='center', rotation='vertical')

fig, ax = plt.subplots(6, 6, figsize=(10, 6))
# Flatten the axs array to access each subplot individually
ax = ax.ravel()

add_plot(6, 0, 0, pro_idoa_month)
add_plot(7, 0, 1, pro_idoa_month)

add_plot(12, 1, 0, pro_idoa_month)
add_plot(13, 1, 1, pro_idoa_month)
add_plot(14, 1, 2, pro_idoa_month)
add_plot(15, 1, 3, pro_idoa_month)

add_plot(18, 2, 0, pro_idoa_month)
add_plot(19, 2, 1, pro_idoa_month)
add_plot(21, 2, 2, pro_idoa_month)
add_plot(22, 2, 3, pro_idoa_month)

add_plot(24, 3, 0, pro_idoa_month)
add_plot(25, 3, 1, pro_idoa_month)
add_plot(26, 3, 2, pro_idoa_month)
add_plot(27, 3, 3, pro_idoa_month)
add_plot(28, 3, 4, pro_idoa_month)

add_plot(30, 4, 0, pro_idoa_month)
add_plot(32, 4, 1, pro_idoa_month)
add_plot(33, 4, 2, pro_idoa_month)
add_plot(34, 4, 3, pro_idoa_month)

for x in ax.flat:
    x.set_xticklabels([])
    x.set_yticklabels([])

for x in ax.flatten():
    x.tick_params(bottom=False, left=False)

fig.text(0.5, 0.04, 'Subjects', ha='center', va='center', fontsize=15)
fig.text(0.06, 0.5, 'Days', ha='center', va='center', rotation='vertical', fontsize=15)

for i, label in enumerate(['1', '2', '3', '4', '5', '7']):
    fig.text(0.18 + i * 0.134, 0.1, label, ha='center', va='center')

for i, label in enumerate(['236', '206', '176', '146', '116', '86']):
    fig.text(0.1, 0.18 + i * 0.125, label, ha='center', va='center', rotation='vertical')

plt.show()

spo_idoa_ant = [idoa_1_day_ant_spo.dissimilarity_overlap_container,
                idoa_2_day_ant_spo.dissimilarity_overlap_container,
                idoa_3_day_ant_spo.dissimilarity_overlap_container,
                idoa_4_day_ant_spo.dissimilarity_overlap_container,
                idoa_5_day_ant_spo.dissimilarity_overlap_container,
                idoa_6_day_ant_spo.dissimilarity_overlap_container,
                idoa_7_day_ant_spo.dissimilarity_overlap_container]

spo_idoa_int = [idoa_1_day_intervension_spo.dissimilarity_overlap_container,
                idoa_2_day_intervension_spo.dissimilarity_overlap_container,
                idoa_3_day_intervension_spo.dissimilarity_overlap_container,
                idoa_4_day_intervension_spo.dissimilarity_overlap_container,
                idoa_5_day_intervension_spo.dissimilarity_overlap_container,
                idoa_6_day_intervension_spo.dissimilarity_overlap_container,
                idoa_7_day_intervension_spo.dissimilarity_overlap_container,
                idoa_14_day_intervension_spo.dissimilarity_overlap_container,
                idoa_21_day_intervension_spo.dissimilarity_overlap_container,
                idoa_28_day_intervension_spo.dissimilarity_overlap_container,
                idoa_42_day_intervension_spo.dissimilarity_overlap_container,
                idoa_56_day_intervension_spo.dissimilarity_overlap_container]

spo_idoa_month = [idoa_month_3_spo.dissimilarity_overlap_container,
                  idoa_month_4_spo.dissimilarity_overlap_container,
                  idoa_month_5_spo.dissimilarity_overlap_container,
                  idoa_month_6_spo.dissimilarity_overlap_container]

fig, ax = plt.subplots(7, 5, figsize=(10, 5))
# Flatten the axs array to access each subplot individually
ax = ax.ravel()


def add_plot(ax_ind, idoa_ind, cont_ind, container):
    slope, intercept = np.polyfit(container[idoa_ind][cont_ind][0, :], container[idoa_ind][cont_ind][1, :], 1)
    x = np.array([container[idoa_ind][cont_ind][0, :].min(), container[idoa_ind][cont_ind][0, :].max()])
    y = slope * x + intercept
    ax[ax_ind].plot(x, y, color='red')
    ax[ax_ind].scatter(container[idoa_ind][cont_ind][0, :], container[idoa_ind][cont_ind][1, :])


add_plot(1, 0, 0, spo_idoa_ant)
add_plot(2, 0, 1, spo_idoa_ant)
add_plot(3, 0, 2, spo_idoa_ant)
add_plot(4, 0, 3, spo_idoa_ant)

add_plot(5, 1, 0, spo_idoa_ant)
add_plot(6, 1, 1, spo_idoa_ant)
add_plot(7, 1, 2, spo_idoa_ant)
add_plot(8, 1, 3, spo_idoa_ant)
add_plot(9, 1, 4, spo_idoa_ant)

add_plot(10, 2, 0, spo_idoa_ant)
add_plot(11, 2, 1, spo_idoa_ant)
add_plot(12, 2, 2, spo_idoa_ant)
add_plot(13, 2, 3, spo_idoa_ant)
add_plot(14, 2, 4, spo_idoa_ant)

add_plot(15, 3, 0, spo_idoa_ant)
add_plot(16, 3, 1, spo_idoa_ant)
add_plot(17, 3, 2, spo_idoa_ant)
add_plot(18, 3, 3, spo_idoa_ant)
add_plot(19, 3, 4, spo_idoa_ant)

add_plot(20, 4, 0, spo_idoa_ant)
add_plot(21, 4, 1, spo_idoa_ant)
add_plot(22, 4, 2, spo_idoa_ant)
add_plot(23, 4, 3, spo_idoa_ant)
add_plot(24, 4, 4, spo_idoa_ant)

add_plot(25, 5, 0, spo_idoa_ant)
add_plot(26, 5, 1, spo_idoa_ant)
add_plot(27, 5, 2, spo_idoa_ant)
add_plot(28, 5, 3, spo_idoa_ant)
add_plot(29, 5, 4, spo_idoa_ant)

add_plot(30, 6, 0, spo_idoa_ant)
add_plot(31, 6, 1, spo_idoa_ant)
add_plot(32, 6, 2, spo_idoa_ant)
add_plot(33, 6, 3, spo_idoa_ant)
add_plot(34, 6, 4, spo_idoa_ant)

for x in ax.flat:
    x.set_xticklabels([])
    x.set_yticklabels([])

for x in ax.flatten():
    x.tick_params(bottom=False, left=False)

fig.text(0.5, 0.004, 'Subjects', ha='center', va='center', fontsize=15)
fig.text(0.06, 0.5, 'Days', ha='center', va='center', rotation='vertical', fontsize=15)

for i, label in enumerate(['1', '2', '3', '4', '5']):
    fig.text(0.18 + i * 0.16, 0.04, label, ha='center', va='center')

for i, label in enumerate(['0', '-1', '-2', '-3', '-4', '-5', '-6']):
    fig.text(0.1, 0.16 + i * 0.11, label, ha='center', va='center', rotation='vertical')

fig, ax = plt.subplots(12, 5, figsize=(10, 10))
# Flatten the axs array to access each subplot individually
ax = ax.ravel()

add_plot(1, 0, 0, spo_idoa_int)
add_plot(2, 0, 1, spo_idoa_int)
add_plot(3, 0, 2, spo_idoa_int)
add_plot(4, 0, 3, spo_idoa_int)

add_plot(6, 1, 0, spo_idoa_int)
add_plot(7, 1, 1, spo_idoa_int)
add_plot(8, 1, 2, spo_idoa_int)
add_plot(9, 1, 3, spo_idoa_int)

add_plot(10, 2, 0, spo_idoa_int)
add_plot(11, 2, 1, spo_idoa_int)
add_plot(12, 2, 2, spo_idoa_int)
add_plot(13, 2, 3, spo_idoa_int)
add_plot(14, 2, 4, spo_idoa_int)

add_plot(15, 3, 0, spo_idoa_int)
add_plot(16, 3, 1, spo_idoa_int)
add_plot(17, 3, 2, spo_idoa_int)
add_plot(18, 3, 3, spo_idoa_int)
add_plot(19, 3, 4, spo_idoa_int)

add_plot(20, 4, 0, spo_idoa_int)
add_plot(21, 4, 1, spo_idoa_int)
add_plot(22, 4, 2, spo_idoa_int)
add_plot(23, 4, 3, spo_idoa_int)
add_plot(24, 4, 4, spo_idoa_int)

add_plot(26, 5, 0, spo_idoa_int)
add_plot(27, 5, 1, spo_idoa_int)
add_plot(28, 5, 2, spo_idoa_int)
add_plot(29, 5, 3, spo_idoa_int)

add_plot(30, 6, 0, spo_idoa_int)
add_plot(31, 6, 1, spo_idoa_int)
add_plot(32, 6, 2, spo_idoa_int)
add_plot(33, 6, 3, spo_idoa_int)
add_plot(34, 6, 4, spo_idoa_int)

add_plot(35, 7, 0, spo_idoa_int)
add_plot(36, 7, 1, spo_idoa_int)
add_plot(37, 7, 2, spo_idoa_int)
add_plot(38, 7, 3, spo_idoa_int)
add_plot(39, 7, 4, spo_idoa_int)

add_plot(40, 8, 0, spo_idoa_int)
add_plot(41, 8, 1, spo_idoa_int)
add_plot(42, 8, 2, spo_idoa_int)
add_plot(43, 8, 3, spo_idoa_int)
add_plot(44, 8, 4, spo_idoa_int)

add_plot(45, 9, 0, spo_idoa_int)
add_plot(46, 9, 1, spo_idoa_int)
add_plot(47, 9, 2, spo_idoa_int)
add_plot(48, 9, 3, spo_idoa_int)
add_plot(49, 9, 4, spo_idoa_int)

add_plot(50, 10, 0, spo_idoa_int)
add_plot(51, 10, 1, spo_idoa_int)
add_plot(52, 10, 2, spo_idoa_int)
add_plot(53, 10, 3, spo_idoa_int)
add_plot(54, 10, 4, spo_idoa_int)

add_plot(55, 11, 0, spo_idoa_int)
add_plot(56, 11, 1, spo_idoa_int)
add_plot(57, 11, 2, spo_idoa_int)
add_plot(58, 11, 3, spo_idoa_int)
add_plot(59, 11, 4, spo_idoa_int)

for x in ax.flat:
    x.set_xticklabels([])
    x.set_yticklabels([])

for x in ax.flatten():
    x.tick_params(bottom=False, left=False)

fig.text(0.5, 0.04, 'Subjects', ha='center', va='center', fontsize=15)
fig.text(0.06, 0.5, 'Days', ha='center', va='center', rotation='vertical', fontsize=15)

for i, label in enumerate(['1', '2', '3', '4', '5']):
    fig.text(0.18 + i * 0.16, 0.1, label, ha='center', va='center')

for i, label in enumerate(['56', '42', '28', '21', '14', '7', '6', '5', '4', '3', '2', '1']):
    fig.text(0.1, 0.16 + i * 0.063, label, ha='center', va='center', rotation='vertical')

fig, ax = plt.subplots(6, 5, figsize=(10, 6))
# Flatten the axs array to access each subplot individually
ax = ax.ravel()

add_plot(10, 0, 0, spo_idoa_month)
add_plot(11, 0, 1, spo_idoa_month)

add_plot(15, 1, 0, spo_idoa_month)
add_plot(16, 1, 1, spo_idoa_month)
add_plot(17, 1, 2, spo_idoa_month)
add_plot(19, 1, 3, spo_idoa_month)

add_plot(20, 2, 0, spo_idoa_month)
add_plot(21, 2, 1, spo_idoa_month)
add_plot(22, 2, 2, spo_idoa_month)
add_plot(24, 2, 3, spo_idoa_month)

add_plot(25, 3, 0, spo_idoa_month)
add_plot(26, 3, 1, spo_idoa_month)
add_plot(27, 3, 2, spo_idoa_month)
add_plot(29, 3, 3, spo_idoa_month)

for x in ax.flat:
    x.set_xticklabels([])
    x.set_yticklabels([])

for x in ax.flatten():
    x.tick_params(bottom=False, left=False)

fig.text(0.5, 0.04, 'Subjects', ha='center', va='center', fontsize=15)
fig.text(0.06, 0.5, 'Days', ha='center', va='center', rotation='vertical', fontsize=15)

for i, label in enumerate(['1', '2', '3', '4', '5']):
    fig.text(0.18 + i * 0.16, 0.1, label, ha='center', va='center')

for i, label in enumerate(['236', '206', '176', '146', '116', '86']):
    fig.text(0.1, 0.18 + i * 0.125, label, ha='center', va='center', rotation='vertical')

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Create a list of vectors
aFMT_BC_int = [BC_1_day_intervension_aFMT_vector, BC_2_day_intervension_aFMT_vector, BC_3_day_intervension_aFMT_vector,
               BC_4_day_intervension_aFMT_vector, BC_5_day_intervension_aFMT_vector, BC_6_day_intervension_aFMT_vector,
               BC_7_day_intervension_aFMT_vector, BC_14_day_intervension_aFMT_vector,
               BC_21_day_intervension_aFMT_vector,
               BC_28_day_intervension_aFMT_vector, BC_42_day_intervension_aFMT_vector,
               BC_56_day_intervension_aFMT_vector]

aFMT_BC_base = [BC_1_day_base_aFMT_vector, BC_2_day_base_aFMT_vector,
                BC_3_day_base_aFMT_vector, BC_4_day_base_aFMT_vector,
                BC_5_day_base_aFMT_vector, BC_6_day_base_aFMT_vector,
                BC_7_day_base_aFMT_vector]

aFMT_BC_ant = [BC_1_day_ant_aFMT_vector, BC_2_day_ant_aFMT_vector,
               BC_3_day_ant_aFMT_vector, BC_4_day_ant_aFMT_vector,
               BC_5_day_ant_aFMT_vector, BC_6_day_ant_aFMT_vector,
               BC_7_day_ant_aFMT_vector]

aFMT_BC_month = [BC_month_1_aFMT_vector, BC_month_2_aFMT_vector, BC_month_2_aFMT_vector,
                 BC_month_4_aFMT_vector, BC_month_5_aFMT_vector, BC_month_6_aFMT_vector]

pro_BC_int = [BC_1_day_intervension_pro_vector, BC_2_day_intervension_pro_vector, BC_3_day_intervension_pro_vector,
              BC_4_day_intervension_pro_vector, BC_5_day_intervension_pro_vector, BC_6_day_intervension_pro_vector,
              BC_7_day_intervension_pro_vector, BC_14_day_intervension_pro_vector, BC_21_day_intervension_pro_vector,
              BC_28_day_intervension_pro_vector, BC_42_day_intervension_pro_vector, BC_56_day_intervension_pro_vector]

pro_BC_base = [BC_1_day_base_pro_vector, BC_2_day_base_pro_vector,
               BC_3_day_base_pro_vector, BC_4_day_base_pro_vector,
               BC_5_day_base_pro_vector, BC_6_day_base_pro_vector,
               BC_7_day_base_pro_vector]

pro_BC_ant = [BC_1_day_ant_pro_vector, BC_2_day_ant_pro_vector,
              BC_3_day_ant_pro_vector, BC_4_day_ant_pro_vector,
              BC_5_day_ant_pro_vector, BC_6_day_ant_pro_vector,
              BC_7_day_ant_pro_vector]

pro_BC_month = [BC_month_2_pro_vector, BC_month_3_pro_vector, BC_month_4_pro_vector,
                BC_month_5_pro_vector, BC_month_6_pro_vector]

spo_BC_int = [BC_1_day_intervension_spo_vector, BC_2_day_intervension_spo_vector, BC_3_day_intervension_spo_vector,
              BC_4_day_intervension_spo_vector, BC_5_day_intervension_spo_vector, BC_6_day_intervension_spo_vector,
              BC_7_day_intervension_spo_vector, BC_14_day_intervension_spo_vector, BC_21_day_intervension_spo_vector,
              BC_28_day_intervension_spo_vector, BC_42_day_intervension_spo_vector, BC_56_day_intervension_spo_vector]

spo_BC_base = [BC_1_day_base_spo_vector, BC_2_day_base_spo_vector,
               BC_3_day_base_spo_vector, BC_4_day_base_spo_vector,
               BC_5_day_base_spo_vector, BC_6_day_base_spo_vector,
               BC_7_day_base_spo_vector]

spo_BC_ant = [BC_1_day_ant_spo_vector, BC_2_day_ant_spo_vector,
              BC_3_day_ant_spo_vector, BC_4_day_ant_spo_vector,
              BC_5_day_ant_spo_vector, BC_6_day_ant_spo_vector,
              BC_7_day_ant_spo_vector]

spo_BC_month = [BC_month_3_spo_vector, BC_month_4_spo_vector,
                BC_month_5_spo_vector, BC_month_6_spo_vector]

# Calculate the mean and standard deviation of each vector
means_aFMT_BC_int = [np.mean(vector) for vector in aFMT_BC_int]
stds_aFMT_BC_int = [np.std(vector) / np.sqrt(len(vector)) for vector in aFMT_BC_int]
means_aFMT_BC_base = [np.mean(vector) for vector in aFMT_BC_base]
stds_aFMT_BC_base = [np.std(vector) / np.sqrt(len(vector)) for vector in aFMT_BC_base]
means_aFMT_BC_ant = [np.mean(vector) for vector in aFMT_BC_ant]
stds_aFMT_BC_ant = [np.std(vector) / np.sqrt(len(vector)) for vector in aFMT_BC_ant]
means_aFMT_BC_month = [np.mean(vector) for vector in aFMT_BC_month]
stds_aFMT_BC_month = [np.std(vector) / np.sqrt(len(vector)) for vector in aFMT_BC_month]

means_pro_BC_int = [np.mean(vector) for vector in pro_BC_int]
stds_pro_BC_int = [np.std(vector) / np.sqrt(len(vector)) for vector in pro_BC_int]
means_pro_BC_base = [np.mean(vector) for vector in pro_BC_base]
stds_pro_BC_base = [np.std(vector) / np.sqrt(len(vector)) for vector in pro_BC_base]
means_pro_BC_ant = [np.mean(vector) for vector in pro_BC_ant]
stds_pro_BC_ant = [np.std(vector) / np.sqrt(len(vector)) for vector in pro_BC_ant]
means_pro_BC_month = [np.mean(vector) for vector in pro_BC_month]
stds_pro_BC_month = [np.std(vector) / np.sqrt(len(vector)) for vector in pro_idoa_month]

means_spo_BC_int = [np.mean(vector) for vector in spo_BC_int]
stds_spo_BC_int = [np.std(vector) / np.sqrt(len(vector)) for vector in spo_BC_int]
means_spo_BC_base = [np.mean(vector) for vector in spo_BC_base]
stds_spo_BC_base = [np.std(vector) / np.sqrt(len(vector)) for vector in spo_BC_base]
means_spo_BC_ant = [np.mean(vector) for vector in spo_BC_ant]
stds_spo_BC_ant = [np.std(vector) / np.sqrt(len(vector)) for vector in spo_BC_ant]
means_spo_BC_month = [np.mean(vector) for vector in spo_BC_month]
stds_spo_BC_month = [np.std(vector) / np.sqrt(len(vector)) for vector in spo_idoa_month]

# Create a list of the number of vectors
days_int = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 42, 56]
days_base = [-13, -12, -11, -10, -9, -8, -7]
days_ant = [-6, -5, -4, -3, -2, -1, 0]
month = [86, 116, 146, 176, 206, 236]
month_no_1 = [116, 146, 176, 206, 236]
month_no_2 = [146, 176, 206, 236]

# Set the figure size
fig, ax = plt.subplots(1, 2, figsize=(18, 7), sharey=True, gridspec_kw={'wspace': 0.1, 'hspace': 0.8})

ax[0].errorbar(days_int, means_aFMT_BC_int, yerr=stds_aFMT_BC_int, fmt='o', capsize=5, ls='none', color='brown')
ax[0].plot(days_int, means_aFMT_BC_int, '-o', color='brown')
ax[0].errorbar(days_base, means_aFMT_BC_base, yerr=stds_aFMT_BC_base, fmt='o', capsize=5, ls='none', color='brown')
ax[0].plot(days_base, means_aFMT_BC_base, '-o', color='brown')
ax[0].errorbar(days_ant, means_aFMT_BC_ant, yerr=stds_aFMT_BC_ant, fmt='o', capsize=5, ls='none', color='brown')
ax[0].plot(days_ant, means_aFMT_BC_ant, '-o', color='brown')

ax[0].errorbar(days_int, means_pro_BC_int, yerr=stds_pro_BC_int, fmt='o', capsize=5, ls='none', color='green')
ax[0].plot(days_int, means_pro_BC_int, '-o', color='green')
ax[0].errorbar(days_base, means_pro_BC_base, yerr=stds_pro_BC_base, fmt='o', capsize=5, ls='none', color='green')
ax[0].plot(days_base, means_pro_BC_base, '-o', color='green')
ax[0].errorbar(days_ant, means_pro_BC_ant, yerr=stds_pro_BC_ant, fmt='o', capsize=5, ls='none', color='green')
ax[0].plot(days_ant, means_pro_BC_ant, '-o', color='green')

ax[0].errorbar(days_int, means_spo_BC_int, yerr=stds_spo_BC_int, fmt='o', capsize=5, ls='none', color='chocolate')
ax[0].plot(days_int, means_spo_BC_int, '-o', color='chocolate')
ax[0].errorbar(days_base, means_spo_BC_base, yerr=stds_spo_BC_base, fmt='o', capsize=5, ls='none', color='chocolate')
ax[0].plot(days_base, means_spo_BC_base, '-o', color='chocolate')
ax[0].errorbar(days_ant, means_spo_BC_ant, yerr=stds_spo_BC_ant, fmt='o', capsize=5, ls='none', color='chocolate')
ax[0].plot(days_ant, means_spo_BC_ant, '-o', color='chocolate')

ax[0].set_ylim(0.4, 1.1)
y_min, y_max = ax[0].get_ylim()
ax[0].set_ylabel('Mean BC', fontsize='20')
ax[0].spines['right'].set_color('white')
ax[0].axvline(x=-7, color='k', linestyle='--')
ax[0].axvline(x=0, color='k', linestyle='--')
ax[0].fill_betweenx([y_min, y_max], -7, 0, color='gray', alpha=0.2)

ax[1].errorbar(month[1:], means_aFMT_BC_month[1:], yerr=stds_aFMT_BC_month[1:], fmt='o', capsize=5, ls='none',
               color='brown')
ax[1].plot(month[1:], means_aFMT_BC_month[1:], '-o', color='brown', label='aFMT')

ax[1].errorbar(month_no_1, means_pro_BC_month, yerr=stds_pro_BC_month, fmt='o', capsize=5, ls='none', color='green')
ax[1].plot(month_no_1, means_pro_BC_month, '-o', color='green', label='Probiotics')

ax[1].errorbar(month_no_2, means_spo_BC_month, yerr=stds_spo_BC_month, fmt='o', capsize=5, ls='none', color='chocolate')
ax[1].plot(month_no_2, means_spo_BC_month, '-o', color='chocolate', label='Spontaneous')

ax[1].set_ylim(0.4, 1.1)
ax[1].get_yaxis().set_visible(False)
ax[1].spines['left'].set_color('white')
ax[1].legend(loc='upper right', fontsize='15')

# Set the x-axis label for both subplots
fig.text(0.5, 0.04, 'Days', ha='center', fontsize='20')

# Add some padding between the plots and the x-axis label
fig.subplots_adjust(bottom=0.15)
plt.show()