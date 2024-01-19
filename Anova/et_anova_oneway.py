import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#anova_filename = "ET_anova_input.csv"
anova_filename = "ET_anova_input_temp.csv"

metrics_list = ['Saccade', 'PupilDiameter', 'LeftPupilDiameter', 'RightPupilDiameter',
                    'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                    'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                    'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                    'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                    'HeadHeading', 'HeadPitch',	'HeadRoll']

metrics_list = ['Saccade', 'PupilDiameter']

anova_df = pd.read_csv(anova_filename, sep=' ', low_memory=False)

for metric in metrics_list:
    
    # One way anova (scenario)
    keys = ['low', 'high', 'medium']

    values = []
    for key in keys:
        values.append(list(anova_df.loc[anova_df['scenario'] == key, metric]))
    
    data = dict(zip(keys, values))

    # stats f_oneway functions takes the groups as input and returns F and P-value
    fvalue, pvalue = stats.f_oneway(data['low'],
                                    data['medium'], 
                                    data['high'])

    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    
    # perform multiple pairwise comparison (Tukey HSD)
    m_comp = pairwise_tukeyhsd(endog=anova_df[metric], groups=anova_df['scenario'], alpha=0.05)
    print(m_comp)
    
    # One way anova (atco_id)
    keys = ['MO', 'EI', 'KV', 'UO', 'KB', 'PF', 'AL', 'IH', 'RI', 'JO', 'AE', 'HC', 'LS', 'ML', 'AP', 'AK', 'RE', 'SV']
    keys = ['RI', 'JO', 'AE', 'HC']

    values = []
    for key in keys:
        values.append(list(anova_df.loc[anova_df['atco_id'] == key, metric]))
    
    data = dict(zip(keys, values))

    # stats f_oneway functions takes the groups as input and returns F and P-value
    '''
    fvalue, pvalue = stats.f_oneway(data['MO'], data['EI'], data['KV'], data['UO'], data['KB'], data['PF'],
                                    data['AL'], data['IH'], data['RI'],  data['JO'], data['AE'], data['HC'],
                                    data['LS'], data['ML'], data['AP'],  data['AK'], data['RE'], data['SV']
                                    )
    '''
    fvalue, pvalue = stats.f_oneway(data['RI'],  data['JO'], data['AE'], data['HC'])

    print(metric)
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    
    # perform multiple pairwise comparison (Tukey HSD)
    m_comp = pairwise_tukeyhsd(endog=anova_df[metric], groups=anova_df['atco_id'], alpha=0.05)
    print(m_comp)
    
    print("\n\n")
