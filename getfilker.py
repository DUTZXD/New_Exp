import requests
import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver
import random
from selenium.webdriver.chrome.options import Options
import re

# http请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3741.400 QQBrowser/10.5.3863.400'
}


def get_proxy():
    ip_list = ['123.207.25.143:3128', '202.85.213.219:3128', '61.4.184.180:3128']
    proxy = urllib.request.ProxyHandler({'http': random.choice(ip_list)})
    opener = urllib.request.build_opener(proxy)
    opener.addheaders = [('User-Agent',
                          'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3741.400 QQBrowser/10.5.3863.400')]
    urllib.request.install_opener(opener)


def browerHtml(url):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.get(url)
    return driver.page_source


def getHtml(ul, code='utf-8'):
    try:
        r = requests.get(ul, headers=headers)
        r.raise_for_status()
        r.encoding = code
        return r.content
    except:
        return ""


def getImgUrl(html):
    soup = BeautifulSoup(html, "lxml")
    patter = r'//[^\s]*.jpg'
    HrefInfo = re.findall(patter, str(soup))
    print(HrefInfo)
    print(len(HrefInfo))
    return HrefInfo


def SaveImg(Lists):
    index = 1
    for ul in Lists:
        url = 'https:' + ul
        Img = getHtml(url)
        print("正在保存第{0}张图片".format(index))
        open('F:\Img_' + str(index) + '.jpg', 'wb').write(Img)
        index = int(index) + 1
    print("图片保存完毕")


def main():
    text = input("请输入你要搜索的图片名称：")
    print(text)
    url = 'https://www.flickr.com/search/?text=' + text + '&view_all=1'
    f = open('Filckr.txt', 'w', encoding='utf-8')
    get_proxy()
    htmlText = browerHtml(url)
    # print(htmlText)
    f.write(htmlText)
    Lists = getImgUrl(htmlText)
    SaveImg(Lists)


if __name__ == '__main__':
    main()
