# MGS-3001-job-crawler Zifei Mo Simon 1337864
This repository includes the crawler code, the data set and some brief description about the data.
Research question: How does the labor market differentially value general-purpose versus specialized AI skills, and how does this vary across industries and experience levels?
In the dataset, we collect several criteria including the job title, title link, salaries, company name, company type, company size and crawling time.
To run this script successfully, you need:
Packages: 
json, time, random, re, pandas, os
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
