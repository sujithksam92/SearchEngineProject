import queue
import re
import requests
import os  
import urllib.robotparser
from urllib.parse import urlparse
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from selenium import webdriver  
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.chrome.options import Options  



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
while(faculty_id<20):
    driver.get(full_url)
    try:
        prof_title = driver.find_elements_by_xpath('//*[@id="page"]/section/div/div/h1')[0]
        prof_desig = driver.find_elements_by_xpath('//*[@id="page"]/section/div/div/div[1]/div/table/tbody/tr/td[2]/div/div/table/tbody/tr[1]/td')[0]
        # print(str(faculty_id)+':'+prof_title.text)
        # print(prof_desig.text)
        # json_file_dict_element['name']=prof_title.text
        json_file_dict_element.update(name = prof_title.text)
        # json_file_dict_element['fc_id']=faculty_id
        json_file_dict_element.update(fc_id = faculty_id)
        # json_file_dict.append(json_file_dict_element)
        json_file_dict_element.update(fc_desig = prof_desig.text)
        # print(json_file_dict_element)
        json_file_dict.append(dict(json_file_dict_element))
    except IndexError:
        pass
    faculty_id+=1
    full_url=core_url+str(faculty_id)

print(json_file_dict)




# rp = urllib.robotparser.RobotFileParser()
# rp.set_url("https://www.stevens.edu/robots.txt")
# rp.read()

# print(rp.can_fetch("*", "https://www.stevens.edu/install.php"))

# email_addresses = []
# checked_urls = []
# q = queue.Queue()
# q.put("https://www.stevens.edu/")
# checked_urls+="https://www.stevens.edu/"

