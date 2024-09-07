import pandas as pd
from langdetect import detect

data = {'text':  ["It is a good option","Better to have this way","es un portal informático para geeks","は、ギーク向けのコンピューターサイエンスポータルです"]}

df = pd.DataFrame(data)

df['lang'] = df['text'].apply(lambda x: detect(x))
print(df['lang'])