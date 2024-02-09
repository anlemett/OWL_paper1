import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

anova_filename = "ET_anova_input.csv"
#anova_filename = "ET_anova_input_temp.csv"

metrics_list = ['Saccade', 'PupilDiameter', 'LeftPupilDiameter', 'RightPupilDiameter',
                    'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                    'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                    'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                    'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                    'HeadHeading', 'HeadPitch',	'HeadRoll']

#metrics_list = ['Saccade', 'PupilDiameter']

anova_df = pd.read_csv(anova_filename, sep=' ')

for metric in metrics_list:

    # Multiple anova (scenario, atco_id)
    formula = metric + ' ~ C(atco_id) + C(scenario)'
    #formula = metric + ' ~ C(atco_id) + C(scenario) + C(atco_id):C(scenario)'
    
    model = ols(formula, data=anova_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)
    print(metric)
    print(anova_table)
        
    print("\n\n")
