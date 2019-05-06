import queue
import re
import requests
import pprint
import os  
import json
import urllib.robotparser
from urllib.parse import urlparse
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from selenium import webdriver  
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.chrome.options import Options  
import numpy as np
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import pyLDAvis
import pyLDAvis.gensim
from sys import platform
import copy
import matplotlib as plt
if platform == "darwin":
    plt.use('TkAgg')    
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib','inline')




# def is_absolute(url):
#     return bool(urlparse(url).netloc)

# chrome_options = Options()  
# chrome_options.add_argument("--headless")  
# driver = webdriver.Chrome('/Users/sujithsam/Downloads/chromedriver',   chrome_options=chrome_options)

options = Options()
options.headless = True
driver = webdriver.Chrome('/Users/sujithsam/Documents/Studies/Stevens/Sem-2/BIS-660-Web-Mining/Research_Engine_Project/chromedriver', chrome_options=options)

#chromedriver = '/Users/sujithsam/Downloads/chromedriver'
#options = webdriver.Chrome(chromedriver)
#options = webdriver.ChromeOptions()
#options.add_argument("headless")

json_file_dict_element={}
json_file_dict=[]
faculty_id=1
core_url="https://web.stevens.edu/facultyprofile/?id="
full_url=core_url+str(faculty_id)
while(faculty_id<3000):
    driver.get(full_url)
    try:
        prof_title = driver.find_elements_by_xpath('//*[@id="page"]/section/div/div/h1')[0]
        prof_desig = driver.find_elements_by_xpath('//*[@id="page"]/section/div/div/div[1]/div/table/tbody/tr/td[2]/div/div/table/tbody/tr[1]/td')[0]
        
        # Print Prof Details to Check
        print(str(faculty_id)+':'+prof_title.text)
        print(prof_desig.text)

        prof_text_li = driver.find_elements_by_tag_name('li')
        # Print LI to check
        for item in prof_text_li:
            print(item.text)
        prof_text_p = driver.find_elements_by_tag_name('p')
        # Print LI to check
        for item in prof_text_p:
            print(item.text)


        # json_file_dict_element['name']=prof_title.text
        json_file_dict_element.update(name = prof_title.text)
        # json_file_dict_element['fc_id']=faculty_id
        json_file_dict_element.update(fc_id = faculty_id)
        # json_file_dict.append(json_file_dict_element)
        json_file_dict_element.update(fc_desig = prof_desig.text)
        # print(json_file_dict_element)
        element_lis={}
        content_list=[]
        for i,item in enumerate(prof_text_li,start=0):
            content_list.insert(i,item.text)
        json_file_dict_element.update(fc_content = content_list)
        for i,item in enumerate(prof_text_p,start=0):
            content_list.insert(i,item.text)
        json_file_dict_element.update(fc_content = content_list)
        # print(content_list)
        # json_file_dict_element.update(fc_li = element_lis)
        
        # for i,item in enumerate(prof_text_p,start=0):
        #     json_file_dict_element.update(fc_content_p[i] = item.text)        
        json_file_dict.append(dict(json_file_dict_element))
    except IndexError:
        pass
    faculty_id+=1
    full_url=core_url+str(faculty_id)

# Pretty Print Start
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(json_file_dict)
# Pretty Print End

#Write JSON File Start
with open('testdata.json','w') as outfile:
    json.dump(json_file_dict,outfile)
#Write JSON File End






# print(json_file_dict)




# rp = urllib.robotparser.RobotFileParser()
# rp.set_url("https://www.stevens.edu/robots.txt")
# rp.read()

# print(rp.can_fetch("*", "https://www.stevens.edu/install.php"))

# email_addresses = []
# checked_urls = []
# q = queue.Queue()
# q.put("https://www.stevens.edu/")
# checked_urls+="https://www.stevens.edu/"

