from tool_chain import create_OpenDataBIM_helper_agent
import pandas as pd
from langchain.llms import OpenAI

df_full = pd.read_csv('./RVT_3000_300_Columns.csv')
llm = OpenAI(temperature=0)

project_name = '12etazhka_888_rvt'
df = df_full[(df_full['Filename'] == project_name) &
        (df_full['Category'] != 'None') &
        (df_full['Category'].apply(lambda x: type(x) == str and x.startswith('OST')))]

propstr = ['Area', 'Volume', 'Depth', 'Width', 'Length', 'Perimeter']

# results in messy df
# for el in propstr:
#     df[el+'_str'] = df[el]

for el in propstr:
    print(el)
    df[el] = df[el].astype(str)
    df[el] = df[el].str.extract('(\d*.?\d*)')
    df[el] = df[el].fillna(0)
    df[el] = df[el].replace(r'n',0, regex=True)
    # df[el] = df[el].apply(lambda x: x.replace(',','.') if type(x)=='str' else x)
    df[el] = df[el].replace(',','.', regex=True).astype(float)

for col in df.columns:
    if (len(df[col].unique()) <= 3 and col != 'Filename') or ('Unnamed' in col):
        del df[col]

agent = create_OpenDataBIM_helper_agent(llm, df, verbose=True)

agent.run("""find the total "Volume" of elements with Category "OST_Walls" it's in cubic meters""")

agent.run("""how much polystyrene concrete cost in Manchester? in euro""")

agent.run("""and how much traditional concrete costs?""")

agent.run("""and how much will it cost to fill the Volume of my project?""")
