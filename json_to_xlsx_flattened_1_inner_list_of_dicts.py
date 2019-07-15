# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:05:46 2019
@author: 1003960

Pakeicia toki formata (kai dict VALUE yra kitas LISTas su dicts viduje):
    [{...:...,
     ...:...,
     ...: [{...:...},
           {...:...}
          ]
     },
     { -||- }
    ]
"""
import pandas
failo_vardas_0 = 'all_loads2018_20190603_1300'
failo_vardas_1 = failo_vardas_0 + '.json'
df=pandas.read_json(failo_vardas_1, encoding = 'utf8')
out_file = failo_vardas_0 + '_' + pandas.to_datetime('today').strftime('%y%m%d%H%M%S') + '.xlsx'
key_kuriame_list = 'load_hours' # vidinis key, kurio value yra tas list'as
key1_vidiniame_list = 'hour_name' # vidiniame list'e esanciu dicts KEY, kurio values eis kaip PARENT i nauju col headers

# loop per df pirmos eilutes (pirmojo dict) vidinio listo PIRMO dict keys, kad rastume KOKIU SAKU reiks kiekvienam vidinio listo dict
saku_list = [] #nauju cols pavadinimu child list
for key, val in df.at[0,key_kuriame_list][0].items():
    if key != key1_vidiniame_list:
        saku_list.append(key)

added_cols_parent_list = [] # nauju cols pavadinimu parent list
# loop per df pirmos eilutes (pirmojo dict) vidinio listo dicts, kad identifikuotume parents jau atrastoms sakoms
for elem in df.at[0,key_kuriame_list]:
    added_cols_parent_list.append(elem.get(key1_vidiniame_list))

# pagaminam nauju col names df + pridedam tuos cols i main df
dfColNames = pandas.DataFrame("",columns = added_cols_parent_list,index = saku_list)
for col_name in added_cols_parent_list:
    for row_label in saku_list:
        dfColNames[col_name][row_label] = col_name + ' _' + row_label
        # pridedam stulpeli i df
        df[dfColNames[col_name][row_label]] = ""

# supildom naujus df stulpelius
# loop per visas df eilutes
for i in range(len(df)):
    # loop per vidinio listo dicts
    for elem in df.at[i,key_kuriame_list]:
        for saka in saku_list:
            # pildom df stulpelius
            df.at[i,dfColNames[elem.get(key1_vidiniame_list)][saka]] = elem.get(saka)

# istrinam col is kurio duomenys jau sukilnoti i naujus cols
df = df.drop(key_kuriame_list, 1)

with pandas.ExcelWriter(out_file) as writer:
    df.to_excel(writer)