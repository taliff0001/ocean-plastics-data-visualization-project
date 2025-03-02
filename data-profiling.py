# Author: Michael S.
# Purpose: Create a ydata profile of finilized dataset

import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv(r'.\outputs\integrated_ocean_plastics.csv')
profile = ProfileReport(df=df, title="microplastics profile")
profile.to_file(r".\outputs\microplastics-profile.html")