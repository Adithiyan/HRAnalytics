import requests
from bs4 import BeautifulSoup

import sys
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import json

from datetime import datetime


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================
class Podcast:
    # Constructor (initializes properties)
    def __init__(self):
        self.title = ""
        self.audioLink = ""
        self.duration = ""
        self.description = ""
        self.transcript = ""

    def display(self):
        print("---")
        print("Podcast Title: " + self.title)
        print("Podcast Audio Link: " + self.audioLink)
        print("Podcast Duration: " + self.duration)
        print("Podcast Description: " + self.description[:30])
        print("Podcast Transcript: " + self.transcript[:30])
        print("---")

    def toDictionary(self):
        outDict = {}
        outDict["podcast_title"] = self.title
        outDict["podcast_audio_link"] = self.audioLink
        outDict["podcast_duration"] = self.duration
        outDict["podcast_description"] = self.description
        outDict["podcast_transcript"] = self.transcript

        return outDict

class PodcastSerie:
    # Constructor (initializes properties)
    def __init__(self):
        self.serieTitle = ""
        self.serieLink = ""
        self.serieDate = ""
        self.serieDescription = ""
        self.serieTagList = []

        # each element of this list is a Podcast object.
        self.podcastList = []
    
    def display(self):
        print("Podcase Serie Title: " + self.serieTitle)
        print("Podcase Serie Date: " + self.serieDate)
        print("Podcase Serie Link: " + self.serieLink)
        print("Podcase Serie Description: " + self.serieDescription)
        print("Podcase Serie Tags: " + str(self.serieTagList))
        print("Podcast Serie contains: " + str(len(self.podcastList)) + " podcasts.")

    def toDictionary(self):
        outDict = {}
        outDict["serie_title"] = self.serieTitle
        outDict["serie_link"] = self.serieLink
        outDict["serie_date"] = self.serieDate
        outDict["serie_description"] = self.serieDescription
        outDict["serie_tags"] = self.serieTagList
        outDict["serie_podcast"] = []

        for item in self.podcastList :
            outDict["serie_podcast"].append(item.toDictionary())

        return outDict
        
# ============================================================================
# FUNCTIONS
# ============================================================================
def parsePodcastPage_singleEP(soup) :

    currParsedPodcast = Podcast()

    # get the title of the current podcast
    head = soup.find('head')

    title = head.find('title')
    currParsedPodcast.title = title.get_text()

    div_class_main = soup.find('main', id='main')
    # get the descrition for the current podcast
    descriptionList = []
    page_intro_wrapper = div_class_main.find('div', class_='page-intro__content-wrapper')
    des_paragraph = page_intro_wrapper.find_all('p')
    for p in des_paragraph :
        descriptionList.append(p.get_text())
    descriptionStr = '\n'.join(descriptionList)
    currParsedPodcast.description = descriptionStr

    # parse the audio section
    sectionObjList = div_class_main.find_all('section')

    for sectionObj in sectionObjList :
        if sectionObj.find('div', class_='audio') :
    
            #print("[DEBUG] parsing 1 audio div class")
    
            # get audio link
            currParsedPodcast.audioLink = sectionObj.find('iframe').get('src')
    
            # get duration
            # duration is nested under the first <p> under the accordion_wrapper
            accordion_wrapper = sectionObj.find('div', class_='accordion__wrapper')
            durationText = accordion_wrapper.find('p').get_text()
            if durationText.startswith("Duration") :
                currParsedPodcast.duration = durationText
    
            # Transcript is nested under all the <p> under the accordion content_wrapper
            accordion_content_wrapper = sectionObj.find('div', class_='accordion__content-wrapper')
            paragraphObjList = accordion_content_wrapper.find_all('p')
            transcriptStrList = []
            for paragraphObj in paragraphObjList :
                transcriptStrList.append(paragraphObj.get_text())
    
            currParsedPodcast.transcript = '\n'.join(transcriptStrList)
    
    return currParsedPodcast
    

# helper function for parsePodcastSerie
# ----------------------------------------------------------------------------
def parsePodcastPage_multipleEPs(inSectionObjList) :

    descriptionList = []
    podcastObjList = []

    for sectionObj in inSectionObjList :
        if sectionObj.find('div', class_='audio') :
    
            #print("[DEBUG] parsing 1 audio div class")
            # Then this section contains the audio link of the podcast
            # This setion should also contain the audio transcript.
            currParsedPodcast = Podcast()
    
            currParsedPodcast.title = sectionObj.find('h2', class_='audio__title').get_text()
            currParsedPodcast.audioLink = sectionObj.find('iframe').get('src')
    
            # duration is nested under the first <p> under the accordion_wrapper
            accordion_wrapper = sectionObj.find('div', class_='accordion__wrapper')
            durationText = accordion_wrapper.find('p').get_text()
            if durationText.startswith("Duration") :
                currParsedPodcast.duration = durationText
    
            # Transcript is nested under all the <p> under the accordion content_wrapper
            accordion_content_wrapper = sectionObj.find('div', class_='accordion__content-wrapper')
            paragraphObjList = accordion_content_wrapper.find_all('p')
            transcriptStrList = []
            for paragraphObj in paragraphObjList :
                transcriptStrList.append(paragraphObj.get_text())
    
            currParsedPodcast.transcript = '\n'.join(transcriptStrList)
    
            podcastObjList.append(currParsedPodcast)
            # debug
            #currParsedPodcast.display()
    
        elif sectionObj.find('div', class_='rich-text__wrapper') :
            # this section should contain all the rich text which is the
            # description of the current podcast.
            #print("[DEBUG] parsing 1 rich-text section")
            pList = sectionObj.find_all('p')
            pStringList = []
            for p in pList:
                pStringList.append(p.get_text())
            descriptionList.append('\n'.join(pStringList))

    # check if the num of parsed description matches the num of parsed podcasts
    if (len(descriptionList) == len(podcastObjList)) :
        # write in the description
        for index, podcast in enumerate(podcastObjList, start=0) :
            podcast.description = descriptionList[index]

    else :
        print("[FATAL] number of parsed description and num of parsed podcast do not match")
        sys.exit()
    

    return podcastObjList

# this function does not return anything, it directly manipulates the input
# object's internal variable self.podcastList
# ----------------------------------------------------------------------------
def parsePodcastSerie(inPodcastSerieObj) :
    print("[DEBUG] Opening Podcase Serie page: " + inPodcastSerieObj.serieLink)
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration (Windows specific)
    chrome_options.add_argument("--no-sandbox")  # Bypass OS security model (Linux specific)
    chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    chrome_options.add_argument("--window-size=1920,1080")  # Set a default window size for headless mode

    driver1 = webdriver.Chrome(options=chrome_options)#"/usr/bin/chromedriver")

    driver1.get(inPodcastSerieObj.serieLink)
    print("[DEBUG] wait 3 sec for the page to render properly")
    time.sleep(2)

    page_source = driver1.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    returnPodcastList = []

    div_class_main = soup.find('main', id='main')
    if div_class_main :

        #print(div_class_main.get_text())
        sectionObjList = div_class_main.find_all('section')

        #print("[DEBUG] found: " + str(len(sectionObjList)) + " sections")

        # Need to check if this page only has one audio <section>, if that is
        # the case, then need to parse the page differently, basically the 
        # page structure is all wrong.
        audioSectionCntr = 0
        for eachSectionObj in sectionObjList :
            if eachSectionObj.find('div', class_='audio') :
                audioSectionCntr = audioSectionCntr + 1

        if (audioSectionCntr > 1) :
            # this page contains multiple podcast episode
            parsedPodcastObjList = parsePodcastPage_multipleEPs(sectionObjList)
            inPodcastSerieObj.podcastList = parsedPodcastObjList

            # debug
            #for podcast in parsedPodcastObjList :
            #    podcast.display()

        elif (audioSectionCntr == 1) :
            # this page contains a single podcast episode
            # in This case need to pass the entire page soup in order to obtain
            # the proper title of the single podcast episode.
            parsedPodcastObj = parsePodcastPage_singleEP(soup)
            inPodcastSerieObj.podcastList.append(parsedPodcastObj)

            # debug
            #parsedPodcastObj.display()

        driver1.close()

    else :
        print("[ERROR] <div class=container> with id=main can not be found!")
        sys.exit()

# ----------------------------------------------------------------------------
def parsePodcastRootPage(soup) :

    allParsedPodcastSerieList = []
    
    element_resultList = soup.find('div', class_='listing__results')
    if element_resultList :
        # find each sub div which is the card
        element_cards = element_resultList.find_all('div', class_='card card--full')
        if (len(element_cards) == 0) :
            print("[ERROR] Could not find any card (podcase serie item).")
            sys.exit()
        else :
            print("[DEBUG] On the current podcast root page, found " + str(len(element_cards)) + " podcast serie item(s)")

            cntr = 1
    
            # each card represents a Podcast Serie which can contain 1 or more episodes.
            for card in element_cards :

                print("[DEBUG] currently parsing: " + str(cntr) + "/" + str(len(element_cards)) + " podcast serie item(s)")
                # each of this card should contain a link that could be a podcast serie.
                currentParsedPodcastSerie = PodcastSerie()
    
                currCardContentHead = card.find('div', class_='card__content-head')
                element_a = currCardContentHead.find('a')
    
                currentParsedPodcastSerie.serieLink  = rootURL + element_a.get('href')
                currentParsedPodcastSerie.serieTitle = element_a.get('title')
    
                currContentDate = card.find('div', class_='card__date')
                currentParsedPodcastSerie.serieDate = currContentDate.get_text()
    
                currContentDescription = card.find('div', class_='card__desc')
                currentParsedPodcastSerie.serieDescription = currContentDescription.get_text()
    
                currContentTags = card.find_all('div', class_='card__tag') 
                for tag in currContentTags :
                    currentParsedPodcastSerie.serieTagList.append(tag.get_text())
    
                # debug print out
                #currentParsedPodcastSerie.display()
                parsePodcastSerie(currentParsedPodcastSerie)
                print("[DEBUG] parsed: " + str(len(currentParsedPodcastSerie.podcastList)) + " episode(s)")
                allParsedPodcastSerieList.append(currentParsedPodcastSerie)

                cntr = cntr + 1
        
    else :
        print("[ERROR] div Element listing_results not found.")
        sys.exit()

    return allParsedPodcastSerieList



# ============================================================================
# MAIN execution loop
# For CIPD Podcast
# Apparently CIPD Podcast uses live rendered page, so request does not actually 
# work.
# ============================================================================
driver = webdriver.Chrome()#("/usr/bin/chromedriver")
rootURL = "https://www.cipd.org"

# assign some dummy current number and end number for the initial iteration.
endNumber = 1
currentLastItemNumber = 10
browsePageNumber = 1
allParsedPodcastSerieList = []

while (currentLastItemNumber != endNumber) :
    url = 'https://www.cipd.org/uk/knowledge/podcasts/?page=' + str(browsePageNumber)
    driver.get(url)
    print("[DEBUG] loading podcast root URL: " + url)
    print("[DEBUG] wait 3 sec for the page to render properly")
    time.sleep(2)

    # reparse the current page.
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # recount the range of the page number
    specific_element = soup.find('div', id='results')
    if specific_element:
        #print(specific_element.get_text())
        pageNumInfo = specific_element.get_text()
    else:
        print("[ERROR] Can not id page range info.")
        sys.exit()

    print("[DEBUG] Currenty parsing page range: " + pageNumInfo)
    # extract page range info
    pageNumInfoList = pageNumInfo.split(" ")
    
    numberOnlyList = []
    for item in pageNumInfoList :
        if item.isdigit() :
            numberOnlyList.append(int(item))
    
    endNumber = numberOnlyList.pop()
    currentLastItemNumber = numberOnlyList.pop()

    browsePageNumber = browsePageNumber + 1

    # Call the parsing mechanism.
    curr_parsedPodcastSerieList = parsePodcastRootPage(soup)
    for parsedPodcastSerie in curr_parsedPodcastSerieList :
        allParsedPodcastSerieList.append(parsedPodcastSerie)

    print("[DEBUG] Parsed: " + str(len(curr_parsedPodcastSerieList)) + " podcast series.")


driver.close()

# Dump the parsed object into a json file.
outList = []
for item in allParsedPodcastSerieList :
    outList.append(item.toDictionary())

now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")

outJsonFileName = "CIPD_Podcast_data__" + formatted_now + ".json" 

with open(outJsonFileName, 'w') as json_file:
    json.dump(outList, json_file, indent=4)
