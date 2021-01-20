# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:47:23 2019
@author: 1003960

Paprastai strukturai:
    [{...:...,
      ...:...,
      ...:...},
     { -||-  },
     { -||-  }
    ]
"""
import pandas
failo_vardas_0 = "pareigybes"
failo_vardas_1 = failo_vardas_0 + ".json"
df=pandas.read_json(failo_vardas_1, encoding = 'utf8')
out_file = failo_vardas_0 + '_' + pandas.to_datetime('today').strftime('%y%m%d%H%M%S') + '.xlsx'

with pandas.ExcelWriter(out_file) as writer:
    df.to_excel(writer)
