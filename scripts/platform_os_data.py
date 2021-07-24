from df_helper import DfHelper

helper = DfHelper()
df = helper.read_csv("../data/AdSmartABdata.csv")
df.drop('browser', inplace=True, axis=1)

helper.save_csv(df, "../data/AdSmartABdata.csv")