# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:05:46 2019
@author: 1003960

Pakeicia toki formata (kai DVIEJU dicts VALUES yra kiti LISTai su dicts viduje):
    [{...:...,
     ...:...,
     ...: [{...:...},
           {...:...}
          ],
     ...: [{...:...},
           {...:...}
          ]
     },
     { -||- }
    ]
"""
import pandas as pd
failo_vardas_0 = 'valandos'
failo_vardas_1 = failo_vardas_0 + '.json'
df=pd.read_json(failo_vardas_1, encoding = 'utf8')
out_file = failo_vardas_0 + '_' + pd.to_datetime('today').strftime('%y%m%d%H%M%S') + '.xlsx'
key_kuriame_list1 = 'load_hours' # vidinis key, kurio value yra tas list'as
key1_vidiniame_list1 = 'hour_name' # vidiniame list'e esanciu dicts KEY, kurio values eis kaip PARENT i nauju col headers
key_kuriame_list2 = 'parent_load_hours'
key1_vidiniame_list2 = 'hour_name'

# loop per df pirmos eilutes (pirmojo dict) vidinio listo PIRMO dict keys, kad rastume KOKIU SAKU reiks kiekvienam vidinio listo dict
saku_list1 = [] #nauju cols pavadinimu child list
for key, val in df.at[0,key_kuriame_list1][0].items():
    if key != key1_vidiniame_list1:
        saku_list1.append(key)
saku_list2 = [] #nauju cols pavadinimu child list
for key, val in df.at[0,key_kuriame_list2][0].items():
    if key != key1_vidiniame_list2:
        saku_list2.append(key)

for i in range(len(df)):
    for dict_elem_in_list in df.at[i,key_kuriame_list1]:
        for saka in saku_list1:
            df.at[i,key_kuriame_list1 + ' ' + dict_elem_in_list.get(key1_vidiniame_list1) + ' _' + saka] = dict_elem_in_list.get(saka)
    for dict_elem_in_list in df.at[i,key_kuriame_list2]:
        for saka in saku_list2:
            df.at[i,key_kuriame_list2 + ' ' + dict_elem_in_list.get(key1_vidiniame_list2) + ' _' + saka] = dict_elem_in_list.get(saka)

# istrinam cols is kuriu duomenys jau sukilnoti i naujus cols
df = df.drop(key_kuriame_list1, 1)
df = df.drop(key_kuriame_list2, 1)


# OPTIONAL REORDERING
first_col_names = ['tab_number',
        'first_name',
        'last_name',
        'full_name',
        'lecturer_faculty_code',
        'lecturer_faculty_name',
        'lecturer_faculty_short_name',
        'study_year',
        'program_code',
        'program_faculty_code',
        'program_faculty_name',
        'program_faculty_short_name',
        'program_level',
        'program_name',
        'program_branch_name',
        'subject_name',
        'subject_type',
        'subject_language',
        'subject_semester',
        'subject_credits',
        'load_id',
        'lecture_parent_load_id',
        'practice_parent_load_id',
        'lecture_program_student_count',
        'lecture_additional_student_count',
        'practice_program_student_count',
        'practice_additional_student_count',
        'lecture_student_count',
        'practice_student_count',
        'total_lecture_hours',
        'total_practice_hours',
        'total_hours']
# actual stulpeliu sarasas
# sort col names ascending
df = df.sort_index(axis=1)
# paimam col names i list
cols = list(df)
# reorder the list of col names
for i in range(len(first_col_names)-1,-1,-1):
    cols.insert(0, cols.pop(cols.index(first_col_names[i])))
# actual reordering of cols in df
df = df.loc[:, cols]


with pd.ExcelWriter(out_file) as writer:
    df.to_excel(writer)
