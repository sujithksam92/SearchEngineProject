import json
import pprint
from collections import OrderedDict
from operator import itemgetter


#See Blank Desig
with open('testdata.json') as f:
    d = json.load(f)
    for i,item in enumerate(d,start=0):
        if item['fc_desig'] == '':
            print(item)
    



#Remove Multiple Deigs & Sort Alphabetically
# unique_elements=set()
# with open('testdata.json') as f:
#     d = json.load(f)
#     for i,item in enumerate(d,start=0):
#         unique_elements.add(item['fc_desig'])
#         # print(item['fc_desig'])

# print(*sorted(unique_elements), sep="\n")