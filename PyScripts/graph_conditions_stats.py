import numpy as np
import scipy.stats as stats
import os.path
import re
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from functools import reduce
#import pdb
import sleepy

#Functionality currently not working or supported:
    # 1) for 24 hour recordings, cannot compare light vs dark phase within the same condition and have the stats graphed against each other.
    # 2) Ranksums not implemented for light vs dark phase


'''
Plot stats for 2 conditions against each other. Will not work with more or less than 2.

Step 1: import graph_condition_stats.py
Step 2: call in ipython like so:
    graph_conditions_stats.plot(ppath, recFile = 'your_data.txt')
                OR    
    graph_conditions_stats.plot(ppath, controlRecordings = [list of filenames as strings], experimentalRecordings = [list of filenames as strings])

@Params:
    ppath -- location of all files and (optional) list of recordings text file
    recordingFile -- name of text file to use for load_dose recordings
    controlRecordings -- can also just run with list of control and experimental recordings instead of text file
    experimentalRecordings -- list of experimental condition recordings
    TwentyFourHourMode -- False: Can run with any length of recordings to compare conditions (eg dreadds)
                          True: for 24 hour recordings to ALSO PLOT LIGHT VS DARK periods
    plotPercents, plotFrequencies, plotDurations -- select which stats to plot
    miceLabels -- select if you want individual mice to be labeled on the graph.
                  True: colored dots labeled by mouse name
                  False: black dots with no individual mouse labels

Will also print out actual stat numbers as well as ranksums statistics in console.
'''
def plot(ppath, controlRecordings=[], experimentalRecordings=[], recordingFile='', TwentyFourHourMode=True,
                          plotPercents=True, plotFrequencies=True, plotDurations=True, miceLabels=True):
    
    
    if recordingFile != '':
        controlRecordings, experimentalRecordingsDict = sleepy.load_dose_recordings(ppath, recordingFile) 
        #load dose recordings returns the control recordings as a list and the experimental recordings as a dicitonary with doses
        #so we have to pull out the recordings associated with their 'dose'
        for key,value in experimentalRecordingsDict.items():
            experimentalRecordings += experimentalRecordingsDict[key]
            

    controlREMPercent = []
    controlWAKEPercent = []
    controlNREMPercent = []
    controlREMDuration = []
    controlWAKEDuration = []
    controlNREMDuration = []
    controlREMFrequency = []
    controlWAKEFrequency = []
    controlNREMFrequency = []
    
    experimentalREMPercent = []
    experimentalWAKEPercent = []
    experimentalNREMPercent = []
    experimentalREMDuration = []
    experimentalWAKEDuration = []
    experimentalNREMDuration = []
    experimentalREMFrequency = []
    experimentalWAKEFrequency = []
    experimentalNREMFrequency = [] 
    
    ### HANDLE CASE WHERE WE'RE COMPARING BETWEEN 2 CONDITIONS ####
    #This should be returning per mouse stats.....
    controlStats, experimentalStats, controlMice, experimentalMice = sleep_ranksums(ppath, controlRecordings, experimentalRecordings, noGraphs=False)
    
    #for each experimental mouse group the individual stats by mouse into a bucket so we can do 2 things:
    #1) average over them to get the bar height and
    #2) plot them individually on top of the bars
    for eachMouse in range(0, len(controlMice)): #for each control mouse
        controlREMPercent.append(controlStats[0][eachMouse][0])
        controlWAKEPercent.append(controlStats[0][eachMouse][1])
        controlNREMPercent.append(controlStats[0][eachMouse][2])
        controlREMDuration.append(controlStats[1][eachMouse][0])
        controlWAKEDuration.append(controlStats[1][eachMouse][1])
        controlNREMDuration.append(controlStats[1][eachMouse][2])
        controlREMFrequency.append(controlStats[2][eachMouse][0])
        controlWAKEFrequency.append(controlStats[2][eachMouse][1])
        controlNREMFrequency.append(controlStats[2][eachMouse][2])
        

    #make the control dictionary
    controlStatsPerMouseFull = {} #this will be a dictionary with mice IDs as keys and each associated value is a list of 9
    mouseIndex=0
    for eachMouse in controlMice:
        if eachMouse in controlStatsPerMouseFull:
            controlStatsPerMouseFull[eachMouse][0].append(controlREMPercent[mouseIndex]) #not each mouse, but index of each mouse
            controlStatsPerMouseFull[eachMouse][1].append(controlWAKEPercent[mouseIndex])
            controlStatsPerMouseFull[eachMouse][2].append(controlNREMPercent[mouseIndex])
            controlStatsPerMouseFull[eachMouse][3].append(controlREMDuration[mouseIndex])
            controlStatsPerMouseFull[eachMouse][4].append(controlWAKEDuration[mouseIndex])
            controlStatsPerMouseFull[eachMouse][5].append(controlNREMDuration[mouseIndex])
            controlStatsPerMouseFull[eachMouse][6].append(controlREMFrequency[mouseIndex])
            controlStatsPerMouseFull[eachMouse][7].append(controlWAKEFrequency[mouseIndex])
            controlStatsPerMouseFull[eachMouse][8].append(controlNREMFrequency[mouseIndex])
            mouseIndex+=1
        else:            
            controlStatsPerMouseFull[eachMouse] = [controlREMPercent[mouseIndex], controlWAKEPercent[mouseIndex], controlNREMPercent[mouseIndex],
                                     controlREMDuration[mouseIndex], controlWAKEDuration[mouseIndex], controlNREMDuration[mouseIndex],
                                     controlREMFrequency[mouseIndex], controlWAKEFrequency[mouseIndex], controlNREMFrequency[mouseIndex]]
            mouseIndex+=1
        
    #Experimental stats
    for eachMouse in range(0,len(experimentalStats[0])): 
        experimentalREMPercent.append(experimentalStats[0][eachMouse][0])
        experimentalWAKEPercent.append(experimentalStats[0][eachMouse][1]) 
        experimentalNREMPercent.append(experimentalStats[0][eachMouse][2])
        experimentalREMDuration.append(experimentalStats[1][eachMouse][0])
        experimentalWAKEDuration.append(experimentalStats[1][eachMouse][1])
        experimentalNREMDuration.append(experimentalStats[1][eachMouse][2])
        experimentalREMFrequency.append(experimentalStats[2][eachMouse][0])
        experimentalWAKEFrequency.append(experimentalStats[2][eachMouse][1])
        experimentalNREMFrequency.append(experimentalStats[2][eachMouse][2])
        
     #make the experimental dictionary
    experimentalStatsPerMouseFull = {} #this will be a dictionary with mice IDs as keys and each associated value is a list of 9
    mouseIndex=0
    for eachMouse in experimentalMice:
        if eachMouse in experimentalStatsPerMouseFull:
            experimentalStatsPerMouseFull[eachMouse][0].append(experimentalREMPercent[mouseIndex]) #not each mouse, but index of each mouse
            experimentalStatsPerMouseFull[eachMouse][1].append(experimentalWAKEPercent[mouseIndex])
            experimentalStatsPerMouseFull[eachMouse][2].append(experimentalNREMPercent[mouseIndex])
            experimentalStatsPerMouseFull[eachMouse][3].append(experimentalREMDuration[mouseIndex])
            experimentalStatsPerMouseFull[eachMouse][4].append(experimentalWAKEDuration[mouseIndex])
            experimentalStatsPerMouseFull[eachMouse][5].append(experimentalNREMDuration[mouseIndex])
            experimentalStatsPerMouseFull[eachMouse][6].append(experimentalREMFrequency[mouseIndex])
            experimentalStatsPerMouseFull[eachMouse][7].append(experimentalWAKEFrequency[mouseIndex])
            experimentalStatsPerMouseFull[eachMouse][8].append(experimentalNREMFrequency[mouseIndex])
            mouseIndex+=1
        else:            
            experimentalStatsPerMouseFull[eachMouse] = [experimentalREMPercent[mouseIndex], experimentalWAKEPercent[mouseIndex], experimentalNREMPercent[mouseIndex],
                                     experimentalREMDuration[mouseIndex], experimentalWAKEDuration[mouseIndex], experimentalNREMDuration[mouseIndex],
                                     experimentalREMFrequency[mouseIndex], experimentalWAKEFrequency[mouseIndex], experimentalNREMFrequency[mouseIndex]]
            mouseIndex+=1
    
    
    if not TwentyFourHourMode:             

        #make color palette
        clrs = sns.color_palette("husl", (len(controlStatsPerMouseFull.keys()) + len(experimentalStatsPerMouseFull.keys())))
    ################## P E R C E N T ######################    
        if plotPercents == True:
            
            #Print Percents
            print('')
            print('Control vs Experimental Percents')
            print([np.mean(controlREMPercent), np.mean(controlWAKEPercent), np.mean(controlNREMPercent)])
            print([np.mean(experimentalREMPercent), np.mean(experimentalWAKEPercent), np.mean(experimentalNREMPercent)])
            
            plt.figure()
            
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlREMPercent), np.mean(controlWAKEPercent), np.mean(controlNREMPercent)], label='Control')
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')       
            barlistExperimental=plt.bar([2,3,5,6,8], height = [np.mean(experimentalREMPercent),0, np.mean(experimentalWAKEPercent),0, np.mean(experimentalNREMPercent)], label='Experimental')
            barlistExperimental[0].set_color('blue')
            barlistExperimental[2].set_color('blue')
            barlistExperimental[4].set_color('blue')
          
            
            #The issue with the colors here is that if different mice are in different conditions then this works since they will always be separate colors.
            #But if we have the same group of mice with different recording days being different conditions then it creates a separately colored dot for the same mouse in the different conditions.
            #SOLVED
            for i in range(len(controlREMPercent)):
                if miceLabels:
                    plt.plot([1], controlREMPercent[i],'o', color=clrs[i], label=controlMice[i])
                else:
                    plt.plot([1], controlREMPercent[i],'o', color='black')
                    
            for i in range(len(experimentalREMPercent)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]: #then it's the same mouse and we don't want to create a new color
                        plt.plot([2], experimentalREMPercent[i],'o', color=clrs[i])
                    else: #if it is a new mouse then we create a new color
                        plt.plot([2], experimentalREMPercent[i],'o', color=clrs[i+len(controlMice)], label=experimentalMice[i])
                else:
                    plt.plot([2], experimentalREMPercent[i],'o', color='black')
                    
            for i in range(len(controlWAKEPercent)):
                if miceLabels:
                    plt.plot([4], controlWAKEPercent[i],'o', color=clrs[i])
                else:
                    plt.plot([4], controlWAKEPercent[i],'o', color='black',)
                
            for i in range(len(experimentalWAKEPercent)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([5], experimentalWAKEPercent[i],'o', color=clrs[i])
                    else:
                        plt.plot([5], experimentalWAKEPercent[i],'o', color=clrs[i+len(controlMice)])
                else:
                    plt.plot([5], experimentalWAKEPercent[i],'o', color='black')
                    
            for i in range(len(controlNREMPercent)):
                if miceLabels:
                    plt.plot([7], controlNREMPercent[i],'o', color=clrs[i])
                else:
                    plt.plot([7], controlNREMPercent[i],'o', color='black')
                    
            for i in range(len(experimentalNREMPercent)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([8], experimentalNREMPercent[i],'o', color=clrs[i])                        
                    else:
                         plt.plot([8], experimentalNREMPercent[i],'o', color=clrs[i+len(controlMice)]) 
                else:
                    plt.plot([8], experimentalNREMPercent[i],'o', color='black') 
                
            plt.legend()
            plt.xticks([1.5, 4.5, 7.5], ['REM', 'WAKE', 'NREM'], rotation=60)
            plt.ylabel('Percentage (%)')
            plt.xlabel('Sleep State')
            plt.title('Percent')
            
            ################## D U R A T I O N S ######################    
        if plotDurations == True:
            
            #Print durations
            print('')
            print('Control vs Experimental Durations')
            print([np.mean(controlREMDuration), np.mean(controlWAKEDuration), np.mean(controlNREMDuration)])
            print([np.mean(experimentalREMDuration), np.mean(experimentalWAKEDuration), np.mean(experimentalNREMDuration)])
            
            plt.figure()
            
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlREMDuration), np.mean(controlWAKEDuration), np.mean(controlNREMDuration)], label='Control')
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')       
            barlistExperimental=plt.bar([2,3,5,6,8], height = [np.mean(experimentalREMDuration),0, np.mean(experimentalWAKEDuration),0, np.mean(experimentalNREMDuration)], label='Experimental')
            barlistExperimental[0].set_color('blue')
            barlistExperimental[2].set_color('blue')
            barlistExperimental[4].set_color('blue')
            
            for i in range(len(controlREMDuration)):
                if miceLabels:
                    plt.plot([1], controlREMDuration[i],'o', color=clrs[i], label=controlMice[i])
                else:
                    plt.plot([1], controlREMDuration[i],'o', color='black')
            for i in range(len(experimentalREMDuration)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalREMDuration[i],'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalREMDuration[i],'o', color=clrs[i+len(controlMice)], label=experimentalMice[i])
                else:
                    plt.plot([2], experimentalREMDuration[i],'o', color='black')
                    
            for i in range(len(controlWAKEDuration)):
                if miceLabels:
                    plt.plot([4], controlWAKEDuration[i],'o', color=clrs[i])
                else:
                    plt.plot([4], controlWAKEDuration[i],'o', color='black',)
                
            for i in range(len(experimentalWAKEDuration)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([5], experimentalWAKEDuration[i],'o', color=clrs[i])
                    else:
                        plt.plot([5], experimentalWAKEDuration[i],'o', color=clrs[i+len(controlMice)])
                else:
                    plt.plot([5], experimentalWAKEDuration[i],'o', color='black')
                    
            for i in range(len(controlNREMDuration)):
                if miceLabels:
                    plt.plot([7], controlNREMDuration[i],'o', color=clrs[i])
                else:
                    plt.plot([7], controlNREMDuration[i],'o', color='black')
                    
            for i in range(len(experimentalNREMDuration)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([8], experimentalNREMDuration[i],'o', color=clrs[i]) 
                    else:
                        plt.plot([8], experimentalNREMDuration[i],'o', color=clrs[i+len(controlMice)]) 
                else:
                    plt.plot([8], experimentalNREMDuration[i],'o', color='black')
            plt.legend() 
            plt.xticks([1.5, 4.5, 7.5], ['REM', 'WAKE', 'NREM'], rotation=60)
            plt.ylabel('Duration (s)')
            plt.xlabel('Sleep State')
            plt.title('Duration')
            
            
        ################## F R E Q U E N C Y ######################    
        if plotFrequencies == True:
            
            print('')
            print('Control vs Experimental Frequencies')
            print([np.mean(controlREMFrequency), np.mean(controlWAKEFrequency), np.mean(controlNREMFrequency)])
            print([np.mean(experimentalREMFrequency), np.mean(experimentalWAKEFrequency), np.mean(experimentalNREMFrequency)])
            
            plt.figure()
            
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlREMFrequency), np.mean(controlWAKEFrequency), np.mean(controlNREMFrequency)], label='Control')
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')       
            barlistExperimental=plt.bar([2,3,5,6,8], height = [np.mean(experimentalREMFrequency),0, np.mean(experimentalWAKEFrequency),0, np.mean(experimentalNREMFrequency)], label='Experimental')
            barlistExperimental[0].set_color('blue')
            barlistExperimental[2].set_color('blue')
            barlistExperimental[4].set_color('blue')
            
            for i in range(len(controlREMFrequency)):
                if miceLabels:
                    plt.plot([1], controlREMFrequency[i],'o', color=clrs[i], label=controlMice[i])
                else:
                    plt.plot([1], controlREMFrequency[i],'o', color='black')
            for i in range(len(experimentalREMFrequency)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalREMFrequency[i],'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalREMFrequency[i],'o', color=clrs[i+len(controlMice)], label=experimentalMice[i])
                else:
                    plt.plot([2], experimentalREMFrequency[i],'o', color='black')
                    
            for i in range(len(controlWAKEFrequency)):
                if miceLabels:
                    plt.plot([4], controlWAKEFrequency[i],'o', color=clrs[i])
                else:
                    plt.plot([4], controlWAKEFrequency[i],'o', color='black',)
                
            for i in range(len(experimentalWAKEFrequency)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([5], experimentalWAKEFrequency[i],'o', color=clrs[i])
                    else:
                        plt.plot([5], experimentalWAKEFrequency[i],'o', color=clrs[i+len(controlMice)])
                else:
                    plt.plot([5], experimentalWAKEFrequency[i],'o', color='black')
                    
            for i in range(len(controlNREMFrequency)):
                if miceLabels:
                    plt.plot([7], controlNREMFrequency[i],'o', color=clrs[i])
                else:
                    plt.plot([7], controlNREMFrequency[i],'o', color='black')
                    
            for i in range(len(experimentalNREMFrequency)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([8], experimentalNREMFrequency[i],'o', color=clrs[i]) 
                    else:
                        plt.plot([8], experimentalNREMFrequency[i],'o', color=clrs[i+len(controlMice)]) 
                else:
                    plt.plot([8], experimentalNREMFrequency[i],'o', color='black')
            plt.legend() 
            plt.xticks([1.5, 4.5, 7.5], ['REM', 'WAKE', 'NREM'], rotation=60)
            plt.ylabel('Frequency')
            plt.xlabel('Sleep State')
            plt.title('Frequency')
            

     ###### HANDLE CASE WHERE WE'RE COMPARING LIGHT AND DARK PHASES #######
    if TwentyFourHourMode == True:  
        ##### PROCESS LIGHT HALF STATS ##### 
        controlStatsPerMouse = {}
        
        for eachRecording in controlRecordings:
            #Get stats for light and dark periods for each recording (must be done individually because each recording has unique tstarts and tends)
            controlLightTemp, controlDarkTemp, mouseID = sleep_half_stats(ppath, eachRecording) #each one of these will be a list of 9 stats, then the mouseID

            #group the stats together by mouse into a dictionary
            if mouseID in controlStatsPerMouse: #if mouseID is a key in stats per mouse
                #add the stats to the 2D list
                #add the new controlLightREMPercent to the first list in the double list of its associated mouse
                controlStatsPerMouse[mouseID][0].append(controlLightTemp[0])
                controlStatsPerMouse[mouseID][1].append(controlLightTemp[1])
                controlStatsPerMouse[mouseID][2].append(controlLightTemp[2])
                controlStatsPerMouse[mouseID][3].append(controlLightTemp[3])
                controlStatsPerMouse[mouseID][4].append(controlLightTemp[4])
                controlStatsPerMouse[mouseID][5].append(controlLightTemp[5])
                controlStatsPerMouse[mouseID][6].append(controlLightTemp[6])
                controlStatsPerMouse[mouseID][7].append(controlLightTemp[7])
                controlStatsPerMouse[mouseID][8].append(controlLightTemp[8])
                controlStatsPerMouse[mouseID][9].append(controlDarkTemp[0])
                controlStatsPerMouse[mouseID][10].append(controlDarkTemp[1])
                controlStatsPerMouse[mouseID][11].append(controlDarkTemp[2])
                controlStatsPerMouse[mouseID][12].append(controlDarkTemp[3])
                controlStatsPerMouse[mouseID][13].append(controlDarkTemp[4])
                controlStatsPerMouse[mouseID][14].append(controlDarkTemp[5])
                controlStatsPerMouse[mouseID][15].append(controlDarkTemp[6])
                controlStatsPerMouse[mouseID][16].append(controlDarkTemp[7])
                controlStatsPerMouse[mouseID][17].append(controlDarkTemp[8])
            else: #create the new double list associated with the mouseID key
                controlStatsPerMouse[mouseID] = [[controlLightTemp[0]],[controlLightTemp[1]],[controlLightTemp[2]],[controlLightTemp[3]]
                ,[controlLightTemp[4]],[controlLightTemp[5]],[controlLightTemp[6]],[controlLightTemp[7]],[controlLightTemp[8]],
                [controlDarkTemp[0]],[controlDarkTemp[1]],[controlDarkTemp[2]],[controlDarkTemp[3]],[controlDarkTemp[4]],
                [controlDarkTemp[5]],[controlDarkTemp[6]],[controlDarkTemp[7]],[controlDarkTemp[8]]]

        #Once stats are grouped together by mouse into a dictionary, average each stat per mouse into a new dictionary to be able to plot the points
        #And average across each stat regardless of mouse to get the actual bar height

        controlStatsPerMouseAverages = {}
        for key, value in controlStatsPerMouse.items():
            newControlStatsList = []
            for eachControlStatsList in value:
                newControlStatsList.append(np.mean(eachControlStatsList)) #append the mean of each of the 9 stats to a new stats list
            controlStatsPerMouseAverages[key] = newControlStatsList #add that new stats list with its mouseID to the new averages dictionary (each one gets appended to the end)
                        
        temp0 = []
        temp1 = []
        temp2 = []
        temp3 = []
        temp4 = []
        temp5 = []
        temp6 = []
        temp7 = []
        temp8 = []
        temp9 = []
        temp10 = []
        temp11 = []
        temp12 = []
        temp13 = []
        temp14 = []
        temp15 = []
        temp16 = []
        temp17 = []
        
        #also average inclusively           
        for key, value in controlStatsPerMouse.items():
            temp0.append(value[0])
            temp1.append(value[1])
            temp2.append(value[2])
            temp3.append(value[3])
            temp4.append(value[4])
            temp5.append(value[5])
            temp6.append(value[6])
            temp7.append(value[7])
            temp8.append(value[8])
            temp9.append(value[9])
            temp10.append(value[10])
            temp11.append(value[11])
            temp12.append(value[12])
            temp13.append(value[13])
            temp14.append(value[14])
            temp15.append(value[15])
            temp16.append(value[16])
            temp17.append(value[17])
        
        controlLightREMPercent = np.mean(temp0)
        controlLightWAKEPercent = np.mean(temp1)
        controlLightNREMPercent = np.mean(temp2)
        controlLightREMDuration = np.mean(temp3)
        controlLightWAKEDuration = np.mean(temp4)
        controlLightNREMDuration = np.mean(temp5)
        controlLightREMFrequency = np.mean(temp6)
        controlLightWAKEFrequency = np.mean(temp7)
        controlLightNREMFrequency = np.mean(temp8)
        
        controlDarkREMPercent = np.mean(temp9)
        controlDarkWAKEPercent = np.mean(temp10)
        controlDarkNREMPercent = np.mean(temp11)
        controlDarkREMDuration = np.mean(temp12)
        controlDarkWAKEDuration = np.mean(temp13)
        controlDarkNREMDuration = np.mean(temp14)
        controlDarkREMFrequency = np.mean(temp15)
        controlDarkWAKEFrequency = np.mean(temp16)
        controlDarkNREMFrequency = np.mean(temp17)   
            
            
        #average across every stat and put them into buckets (to plot the average across mice)
           
        ##### PROCESS DARK HALF STATS #####    
        experimentalStatsPerMouse = {}
        
        for eachRecording in experimentalRecordings:
            #Get stats for light and dark periods for each recording (must be done individually because each recording has unique tstarts and tends)
            experimentalLightTemp, experimentalDarkTemp, mouseID = sleep_half_stats(ppath, eachRecording) #each one of these will be a list of 9 stats, then the mouseID

            #group the stats together by mouse into a dictionary
            if mouseID in experimentalStatsPerMouse: #if mouseID is a key in stats per mouse
                #add the stats to the 2D list
                #add the new controlLightREMPercent to the first list in the double list of its associated mouse
                experimentalStatsPerMouse[mouseID][0].append(experimentalLightTemp[0])
                experimentalStatsPerMouse[mouseID][1].append(experimentalLightTemp[1])
                experimentalStatsPerMouse[mouseID][2].append(experimentalLightTemp[2])
                experimentalStatsPerMouse[mouseID][3].append(experimentalLightTemp[3])
                experimentalStatsPerMouse[mouseID][4].append(experimentalLightTemp[4])
                experimentalStatsPerMouse[mouseID][5].append(experimentalLightTemp[5])
                experimentalStatsPerMouse[mouseID][6].append(experimentalLightTemp[6])
                experimentalStatsPerMouse[mouseID][7].append(experimentalLightTemp[7])
                experimentalStatsPerMouse[mouseID][8].append(experimentalLightTemp[8])
                experimentalStatsPerMouse[mouseID][9].append(experimentalDarkTemp[0])
                experimentalStatsPerMouse[mouseID][10].append(experimentalDarkTemp[1])
                experimentalStatsPerMouse[mouseID][11].append(experimentalDarkTemp[2])
                experimentalStatsPerMouse[mouseID][12].append(experimentalDarkTemp[3])
                experimentalStatsPerMouse[mouseID][13].append(experimentalDarkTemp[4])
                experimentalStatsPerMouse[mouseID][14].append(experimentalDarkTemp[5])
                experimentalStatsPerMouse[mouseID][15].append(experimentalDarkTemp[6])
                experimentalStatsPerMouse[mouseID][16].append(experimentalDarkTemp[7])
                experimentalStatsPerMouse[mouseID][17].append(experimentalDarkTemp[8])
            else: #create the new double list associated with the mouseID key
                experimentalStatsPerMouse[mouseID] = [[experimentalLightTemp[0]],[experimentalLightTemp[1]],[experimentalLightTemp[2]],[experimentalLightTemp[3]]
                ,[experimentalLightTemp[4]],[experimentalLightTemp[5]],[experimentalLightTemp[6]],[experimentalLightTemp[7]],[experimentalLightTemp[8]],
                [experimentalDarkTemp[0]],[experimentalDarkTemp[1]],[experimentalDarkTemp[2]],[experimentalDarkTemp[3]],[experimentalDarkTemp[4]],
                [experimentalDarkTemp[5]],[experimentalDarkTemp[6]],[experimentalDarkTemp[7]],[experimentalDarkTemp[8]]]

        #Once stats are grouped together by mouse into a dictionary, average each stat per mouse into a new dictionary to be able to plot the points
        #And average across each stat regardless of mouse to get the actual bar height

        experimentalStatsPerMouseAverages = {}
        for key, value in experimentalStatsPerMouse.items():
            newExperimentalStatsList = []
            for eachExperimentalStatsList in value:
                newExperimentalStatsList.append(np.mean(eachExperimentalStatsList)) #append the mean of each of the 9 stats to a new stats list
            experimentalStatsPerMouseAverages[key] = newExperimentalStatsList
#            pdb.set_trace()#add that new stats list with its mouseID to the new averages dictionary (each one gets appended to the end)
                        
        tempB0 = []
        tempB1 = []
        tempB2 = []
        tempB3 = []
        tempB4 = []
        tempB5 = []
        tempB6 = []
        tempB7 = []
        tempB8 = []
        tempB9 = []
        tempB10 = []
        tempB11 = []
        tempB12 = []
        tempB13 = []
        tempB14 = []
        tempB15 = []
        tempB16 = []
        tempB17 = []
        
        #also average inclusively           
        for key, value in experimentalStatsPerMouse.items():
            tempB0.append(value[0]) #sometimes appending a list instead of each value in the list
            tempB1.append(value[1])
            tempB2.append(value[2])
            tempB3.append(value[3])
            tempB4.append(value[4])
            tempB5.append(value[5])
            tempB6.append(value[6])
            tempB7.append(value[7])
            tempB8.append(value[8])
            tempB9.append(value[9])
            tempB10.append(value[10])
            tempB11.append(value[11])
            tempB12.append(value[12])
            tempB13.append(value[13])
            tempB14.append(value[14])
            tempB15.append(value[15])
            tempB16.append(value[16])
            tempB17.append(value[17])
             
        test0 = []
        for x in tempB0:
            for y in x:
                test0.append(y)
        
        test1 = []
        for x in tempB1:
            for y in x:
                test1.append(y)
                
        test2 = []
        for x in tempB2:
            for y in x:
                test2.append(y)
                
        test3 = []
        for x in tempB3:
            for y in x:
                test3.append(y)
                
        test4 = []
        for x in tempB4:
            for y in x:
                test4.append(y)
                
        test5 = []
        for x in tempB5:
            for y in x:
                test5.append(y)
                
        test6 = []
        for x in tempB6:
            for y in x:
                test6.append(y)
                
        test7 = []
        for x in tempB7:
            for y in x:
                test7.append(y)
                
        test8 = []
        for x in tempB8:
            for y in x:
                test8.append(y)
                
        test9 = []
        for x in tempB9:
            for y in x:
                test9.append(y)
                
        test10 = []
        for x in tempB10:
            for y in x:
                test10.append(y)
                
        test11 = []
        for x in tempB11:
            for y in x:
                test11.append(y)
                
        test12 = []
        for x in tempB12:
            for y in x:
                test12.append(y)
                
        test13 = []
        for x in tempB13:
            for y in x:
                test13.append(y)
                
        test14 = []
        for x in tempB14:
            for y in x:
                test14.append(y)
                
        test15 = []
        for x in tempB15:
            for y in x:
                test15.append(y)
                
        test16 = []
        for x in tempB16:
            for y in x:
                test16.append(y)
                
        test17 = []
        for x in tempB17:
            for y in x:
                test17.append(y)
  

        experimentalLightREMPercent = np.mean(test0)     
        experimentalLightWAKEPercent = np.mean(test1)
        experimentalLightNREMPercent = np.mean(test2)
        experimentalLightREMDuration = np.mean(test3)
        experimentalLightWAKEDuration = np.mean(test4)
        experimentalLightNREMDuration = np.mean(test5)
        experimentalLightREMFrequency = np.mean(test6)
        experimentalLightWAKEFrequency = np.mean(test7)
        experimentalLightNREMFrequency = np.mean(test8)
        
        experimentalDarkREMPercent = np.mean(test9)
        experimentalDarkWAKEPercent = np.mean(test10)
        experimentalDarkNREMPercent = np.mean(test11)
        experimentalDarkREMDuration = np.mean(test12)
        experimentalDarkWAKEDuration = np.mean(test13)
        experimentalDarkNREMDuration = np.mean(test14)
        experimentalDarkREMFrequency = np.mean(test15)
        experimentalDarkWAKEFrequency = np.mean(test16)
        experimentalDarkNREMFrequency = np.mean(test17)   
        
        
        clrs = sns.color_palette("husl", (len(controlStatsPerMouseAverages.keys()) + len(experimentalStatsPerMouse.keys())))
               
    ################## P E R C E N T ######################    
        if plotPercents == True:
            
            #Print percentages to compare
            print('')
            print('Full Recording, Control vs Experimental Percents')
            print([np.mean(controlREMPercent), np.mean(controlWAKEPercent), np.mean(controlNREMPercent)])
            print([np.mean(experimentalREMPercent), np.mean(experimentalWAKEPercent), np.mean(experimentalNREMPercent)])
            print('Light Period Percents, Control vs Experimental')
            print([controlLightREMPercent, controlLightWAKEPercent, controlLightNREMPercent])
            print([experimentalLightREMPercent, experimentalLightWAKEPercent, experimentalLightNREMPercent])
            print('Dark Period Percents, Control vs Experimental')
            print([controlDarkREMPercent, controlDarkWAKEPercent, controlDarkNREMPercent])
            print([experimentalDarkREMPercent, experimentalDarkWAKEPercent, experimentalDarkNREMPercent])
            
            #######REM PERCENT########
            plt.figure()      
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlREMPercent), controlLightREMPercent, controlDarkREMPercent], label='Control')             
            barlistExperimental=plt.bar([2,5,8], height = [np.mean(experimentalREMPercent), experimentalLightREMPercent, experimentalDarkREMPercent], label='Experimental')
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')
            barlistExperimental[0].set_color('b')
            barlistExperimental[1].set_color('b')
            barlistExperimental[2].set_color('b')                     
            #Plot individual mouse points for full recording control bar            
            for i in range(len(controlREMPercent)):
                if miceLabels:
                    plt.plot([1], controlREMPercent[i], 'o', color=clrs[i])
                else:                   
                    plt.plot([1], controlREMPercent[i], 'o', color='black')
            #Plot individual mouse points for full recording experimental bar     
            for i in range(len(experimentalREMPercent)):
                if miceLabels: 
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalREMPercent[i], 'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalREMPercent[i], 'o', color=clrs[i+ len(experimentalStatsPerMouse.keys())])
                else:                   
                    plt.plot([2], experimentalREMPercent[i], 'o', color='black')
                
            #Print individual mouse points in control columns for light and dark bars
            colornum=0
            for key, value in controlStatsPerMouseAverages.items():
                if miceLabels:
                    plt.plot([4,7], [value[0], value[9]], 'o', label=key, color=clrs[colornum])
                    colornum+=1
                else:
                    plt.plot([4,7], [value[0], value[9]], 'o', color='black')
                    
            #Print individual mouse points in experimental columns for light and dark bars
            colornum=0
            for key, value in experimentalStatsPerMouseAverages.items():
                if miceLabels:
                    if experimentalMice[colornum] == controlMice[colornum]:
                        plt.plot([5,8], [value[0], value[9]], 'o', color=clrs[colornum])
                    else:
                        plt.plot([5,8], [value[0], value[9]], 'o', label=key, color=clrs[colornum + len(experimentalStatsPerMouse.keys())])
                    colornum+=1
                else:
                    plt.plot([5,8], [value[0], value[9]], 'o', color='black')
                                
            plt.legend()         
            plt.ylim(0,15)   
            plt.xticks([1.5, 4.5, 7.5], ['24', 'Light', 'Dark'], rotation=60)
            plt.ylabel('Percentage (%)')
            plt.xlabel('Condition')
            plt.title('REM Percents')
            
            #######WAKE PERCENT########
            plt.figure()
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlWAKEPercent), controlLightWAKEPercent, controlDarkWAKEPercent], label='Control')   
            barlistExperimental=plt.bar([2,5,8], height = [np.mean(experimentalWAKEPercent), experimentalLightWAKEPercent, experimentalDarkWAKEPercent], label='Experimental')
            #color bars, comes after bars but before legend, legend then comes before point plot to keep them out of it
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')
            barlistExperimental[0].set_color('b')
            barlistExperimental[1].set_color('b')
            barlistExperimental[2].set_color('b')
            #Plot individual mouse points for full recording control bar            
            for i in range(len(controlWAKEPercent)):
                if miceLabels:
                    plt.plot([1], controlWAKEPercent[i], 'o', color=clrs[i])
                else:                   
                    plt.plot([1], controlWAKEPercent[i], 'o', color='black')
            #Plot individual mouse points for full recording experimental bar     
            for i in range(len(experimentalWAKEPercent)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalWAKEPercent[i], 'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalWAKEPercent[i], 'o', color=clrs[i+ len(experimentalStatsPerMouse.keys())])
                else:                   
                    plt.plot([2], experimentalWAKEPercent[i], 'o', color='black')
                
            #Print individual mouse points in control columns for light and dark bars
            colornum=0
            for key, value in controlStatsPerMouseAverages.items():
                if miceLabels:
                    plt.plot([4,7], [value[1], value[10]], 'o', label=key, color=clrs[colornum])
                    colornum+=1
                else:
                    plt.plot([4,7], [value[1], value[10]], 'o', color='black')
                    
            #Print individual mouse points in experimental columns for light and dark bars
            colornum=0
            for key, value in experimentalStatsPerMouseAverages.items():
                if miceLabels:
                    if experimentalMice[colornum] == controlMice[colornum]:
                        plt.plot([5,8], [value[1], value[10]], 'o', color=clrs[colornum])
                    else:
                        plt.plot([5,8], [value[1], value[10]], 'o', label=key, color=clrs[colornum + len(experimentalStatsPerMouse.keys())])
                    colornum+=1
                else:
                    plt.plot([5,8], [value[1], value[10]], 'o', color='black')                                
            plt.legend()                         
            plt.ylim(0,75)   
            plt.xticks([1.5, 4.5, 7.5], ['24', 'Light', 'Dark'], rotation=60)
            plt.ylabel('Percentage (%)')
            plt.xlabel('Condition')
            plt.title('WAKE Percents')
            
            #######NREM PERCENT########
            plt.figure()
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlNREMPercent), controlLightNREMPercent, controlDarkNREMPercent], label='Control')   
            barlistExperimental=plt.bar([2,5,8], height = [np.mean(experimentalNREMPercent), experimentalLightNREMPercent, experimentalDarkNREMPercent], label='Experimental')
            #color bars, comes after bars but before legend, legend then comes before point plot to keep them out of it
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')
            barlistExperimental[0].set_color('b')
            barlistExperimental[1].set_color('b')
            barlistExperimental[2].set_color('b')
            #Plot individual mouse points for full recording control bar            
            for i in range(len(controlNREMPercent)):
                if miceLabels:
                    plt.plot([1], controlNREMPercent[i], 'o', color=clrs[i])
                else:                   
                    plt.plot([1], controlNREMPercent[i], 'o', color='black')
            #Plot individual mouse points for full recording experimental bar     
            for i in range(len(experimentalNREMPercent)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalNREMPercent[i], 'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalNREMPercent[i], 'o', color=clrs[i+ len(experimentalStatsPerMouse.keys())])
                else:                   
                    plt.plot([2], experimentalNREMPercent[i], 'o', color='black')
                
            #Print individual mouse points in control columns for light and dark bars
            colornum=0
            for key, value in controlStatsPerMouseAverages.items():
                if miceLabels:
                    plt.plot([4,7], [value[2], value[11]], 'o', label=key, color=clrs[colornum])
                    colornum+=1
                else:
                    plt.plot([4,7], [value[2], value[11]], 'o', color='black')
                    
            #Print individual mouse points in experimental columns for light and dark bars
            colornum=0
            for key, value in experimentalStatsPerMouseAverages.items():
                if miceLabels:
                    if experimentalMice[colornum] == controlMice[colornum]:
                        plt.plot([5,8], [value[2], value[11]], 'o', color=clrs[colornum])
                    else:
                        plt.plot([5,8], [value[2], value[11]], 'o', label=key, color=clrs[colornum + len(experimentalStatsPerMouse.keys())])
                    colornum+=1
                else:
                    plt.plot([5,8], [value[2], value[11]], 'o', color='black')                                
            plt.legend()    
            plt.ylim(0,65)   
            plt.xticks([1.5, 4.5, 7.5], ['24', 'Light', 'Dark'], rotation=60)
            plt.ylabel('Percentage (%)')
            plt.xlabel('Condition')
            plt.title('NREM Percents')
            
            
    ################## D U R A T I O N ######################
        if plotDurations == True:
            #Print durations to compare
            print('')
            print('Full Recording, Control vs Experimental Durations')
            print([np.mean(controlREMDuration), np.mean(controlWAKEDuration), np.mean(controlNREMDuration)])
            print([np.mean(experimentalREMDuration), np.mean(experimentalWAKEDuration), np.mean(experimentalNREMDuration)])
            print('Light Period Durations, Control vs Experimental')
            print([controlLightREMDuration, controlLightWAKEDuration, controlLightNREMDuration])
            print([experimentalLightREMDuration, experimentalLightWAKEDuration, experimentalLightNREMDuration])
            print('Dark Period Durations, Control vs Experimental')
            print([controlDarkREMDuration, controlDarkWAKEDuration, controlDarkNREMDuration])
            print([experimentalDarkREMDuration, experimentalDarkWAKEDuration, experimentalDarkNREMDuration])
            
            #######REM DURATION########
            plt.figure()
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlREMDuration), controlLightREMDuration, controlDarkREMDuration], label='Control')   
            barlistExperimental=plt.bar([2,5,8], height = [np.mean(experimentalREMDuration), experimentalLightREMDuration, experimentalDarkREMDuration], label='Experimental')
            #color bars, comes after bars but before legend, legend then comes before point plot to keep them out of it
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')
            barlistExperimental[0].set_color('b')
            barlistExperimental[1].set_color('b')
            barlistExperimental[2].set_color('b')
            #Plot individual mouse points for full recording control bar            
            for i in range(len(controlREMDuration)):
                if miceLabels:
                    plt.plot([1], controlREMDuration[i], 'o', color=clrs[i])
                else:                   
                    plt.plot([1], controlREMDuration[i], 'o', color='black')
            #Plot individual mouse points for full recording experimental bar     
            for i in range(len(experimentalREMDuration)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalREMDuration[i], 'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalREMDuration[i], 'o', color=clrs[i+ len(experimentalStatsPerMouse.keys())])
                else:                   
                    plt.plot([2], experimentalREMDuration[i], 'o', color='black')
                
            #Print individual mouse points in control columns for light and dark bars
            colornum=0
            for key, value in controlStatsPerMouseAverages.items():
                if miceLabels:
                    plt.plot([4,7], [value[3], value[12]], 'o', label=key, color=clrs[colornum])
                    colornum+=1
                else:
                    plt.plot([4,7], [value[3], value[12]], 'o', color='black')
                    
            #Print individual mouse points in experimental columns for light and dark bars
            colornum=0
            for key, value in experimentalStatsPerMouseAverages.items():
                if miceLabels:
                    if experimentalMice[colornum] == controlMice[colornum]:
                        plt.plot([5,8], [value[3], value[12]], 'o', color=clrs[colornum])
                    else:
                        plt.plot([5,8], [value[3], value[12]], 'o', label=key, color=clrs[colornum + len(experimentalStatsPerMouse.keys())])
                    colornum+=1
                else:
                    plt.plot([5,8], [value[3], value[12]], 'o', color='black')                                
            plt.legend()    
            #plt.ylim(0,15)   
            plt.xticks([1.5, 4.5, 7.5], ['24', 'Light', 'Dark'], rotation=60)
            plt.ylabel('Duration (s)')
            plt.xlabel('Condition')
            plt.title('REM Durations')
            
            #######WAKE Duration########
            plt.figure()
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlWAKEDuration), controlLightWAKEDuration, controlDarkWAKEDuration], label='Control')   
            barlistExperimental=plt.bar([2,5,8], height = [np.mean(experimentalWAKEDuration), experimentalLightWAKEDuration, experimentalDarkWAKEDuration], label='Experimental')
            #color bars, comes after bars but before legend, legend then comes before point plot to keep them out of it
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')
            barlistExperimental[0].set_color('b')
            barlistExperimental[1].set_color('b')
            barlistExperimental[2].set_color('b')
            #Plot individual mouse points for full recording control bar            
            for i in range(len(controlWAKEDuration)):
                if miceLabels:
                    plt.plot([1], controlWAKEDuration[i], 'o', color=clrs[i])
                else:                   
                    plt.plot([1], controlWAKEDuration[i], 'o', color='black')
            #Plot individual mouse points for full recording experimental bar     
            for i in range(len(experimentalWAKEDuration)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalWAKEDuration[i], 'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalWAKEDuration[i], 'o', color=clrs[i+ len(experimentalStatsPerMouse.keys())])
                else:                   
                    plt.plot([2], experimentalWAKEDuration[i], 'o', color='black')
                
            #Print individual mouse points in control columns for light and dark bars
            colornum=0
            for key, value in controlStatsPerMouseAverages.items():
                if miceLabels:
                    plt.plot([4,7], [value[4], value[13]], 'o', label=key, color=clrs[colornum])
                    colornum+=1
                else:
                    plt.plot([4,7], [value[4], value[13]], 'o', color='black')
                    
            #Print individual mouse points in experimental columns for light and dark bars
            colornum=0
            for key, value in experimentalStatsPerMouseAverages.items():
                if miceLabels:
                    if experimentalMice[colornum] == controlMice[colornum]:
                        plt.plot([5,8], [value[4], value[13]], 'o', color=clrs[colornum])
                    else:
                        plt.plot([5,8], [value[4], value[13]], 'o', label=key, color=clrs[colornum + len(experimentalStatsPerMouse.keys())])
                    colornum+=1
                else:
                    plt.plot([5,8], [value[4], value[13]], 'o', color='black')                                
            plt.legend()    
            #plt.ylim(0,15)   
            plt.xticks([1.5, 4.5, 7.5], ['24', 'Light', 'Dark'], rotation=60)
            plt.ylabel('Duration (s)')
            plt.xlabel('Condition')
            plt.title('WAKE Durations')
            
            #######NREM Duration########
            plt.figure()
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlNREMDuration), controlLightNREMDuration, controlDarkNREMDuration], label='Control')   
            barlistExperimental=plt.bar([2,5,8], height = [np.mean(experimentalNREMDuration), experimentalLightNREMDuration, experimentalDarkNREMDuration], label='Experimental')
            #color bars, comes after bars but before legend, legend then comes before point plot to keep them out of it
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')
            barlistExperimental[0].set_color('b')
            barlistExperimental[1].set_color('b')
            barlistExperimental[2].set_color('b')
            #Plot individual mouse points for full recording control bar            
            for i in range(len(controlNREMDuration)):
                if miceLabels:
                    plt.plot([1], controlNREMDuration[i], 'o', color=clrs[i])
                else:                   
                    plt.plot([1], controlNREMDuration[i], 'o', color='black')
            #Plot individual mouse points for full recording experimental bar     
            for i in range(len(experimentalNREMDuration)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalNREMDuration[i], 'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalNREMDuration[i], 'o', color=clrs[i+ len(experimentalStatsPerMouse.keys())])
                else:                   
                    plt.plot([2], experimentalNREMDuration[i], 'o', color='black')
                
            #Print individual mouse points in control columns for light and dark bars
            colornum=0
            for key, value in controlStatsPerMouseAverages.items():
                if miceLabels:
                    plt.plot([4,7], [value[5], value[14]], 'o', label=key, color=clrs[colornum])
                    colornum+=1
                else:
                    plt.plot([4,7], [value[5], value[14]], 'o', color='black')
                    
            #Print individual mouse points in experimental columns for light and dark bars
            colornum=0
            for key, value in experimentalStatsPerMouseAverages.items():
                if miceLabels:
                    if experimentalMice[colornum] == controlMice[colornum]:
                        plt.plot([5,8], [value[5], value[14]], 'o', color=clrs[colornum])
                    else:
                        plt.plot([5,8], [value[5], value[14]], 'o', label=key, color=clrs[colornum + len(experimentalStatsPerMouse.keys())])
                    colornum+=1
                else:
                    plt.plot([5,8], [value[5], value[14]], 'o', color='black')                                
            plt.legend()    
            #plt.ylim(0,15)   
            plt.xticks([1.5, 4.5, 7.5], ['24', 'Light', 'Dark'], rotation=60)
            plt.ylabel('Duration (s)')
            plt.xlabel('Condition')
            plt.title('NREM Durations')
        
    
    
    ################## F R E Q U E N C Y ######################
        if plotFrequencies == True:
            #Print frequencies to compare
            print('')
            print('Full Recording, Control vs Experimental Frequencies')
            print([np.mean(controlREMFrequency), np.mean(controlWAKEFrequency), np.mean(controlNREMFrequency)])
            print([np.mean(experimentalREMFrequency), np.mean(experimentalWAKEFrequency), np.mean(experimentalNREMFrequency)])
            print('Light Period Frequencies, Control vs Experimental')
            print([controlLightREMFrequency, controlLightWAKEFrequency, controlLightNREMFrequency])
            print([experimentalLightREMFrequency, experimentalLightWAKEFrequency, experimentalLightNREMFrequency])
            print('Dark Period Frequencies, Control vs Experimental')
            print([controlDarkREMFrequency, controlDarkWAKEFrequency, controlDarkNREMFrequency])
            print([experimentalDarkREMFrequency, experimentalDarkWAKEFrequency, experimentalDarkNREMFrequency])
            #######REM FREQUENCY########
            plt.figure()
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlREMFrequency), controlLightREMFrequency, controlDarkREMFrequency], label='Control')   
            barlistExperimental=plt.bar([2,5,8], height = [np.mean(experimentalREMFrequency), experimentalLightREMFrequency, experimentalDarkREMFrequency], label='Experimental')
            #color bars, comes after bars but before legend, legend then comes before point plot to keep them out of it
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')
            barlistExperimental[0].set_color('b')
            barlistExperimental[1].set_color('b')
            barlistExperimental[2].set_color('b')
            #Plot individual mouse points for full recording control bar            
            for i in range(len(controlREMFrequency)):
                if miceLabels:
                    plt.plot([1], controlREMFrequency[i], 'o', color=clrs[i])
                else:                   
                    plt.plot([1], controlREMFrequency[i], 'o', color='black')
            #Plot individual mouse points for full recording experimental bar     
            for i in range(len(experimentalREMFrequency)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalREMFrequency[i], 'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalREMFrequency[i], 'o', color=clrs[i+ len(experimentalStatsPerMouse.keys())])
                else:                   
                    plt.plot([2], experimentalREMFrequency[i], 'o', color='black')
                
            #Print individual mouse points in control columns for light and dark bars
            colornum=0
            for key, value in controlStatsPerMouseAverages.items():
                if miceLabels:
                    plt.plot([4,7], [value[6], value[15]], 'o', label=key, color=clrs[colornum])
                    colornum+=1
                else:
                    plt.plot([4,7], [value[6], value[15]], 'o', color='black')
                    
            #Print individual mouse points in experimental columns for light and dark bars
            colornum=0
            for key, value in experimentalStatsPerMouseAverages.items():
                if miceLabels:
                    if experimentalMice[colornum] == controlMice[colornum]:
                        plt.plot([5,8], [value[6], value[15]], 'o', label=key, color=clrs[colornum])                        
                    else:
                        plt.plot([5,8], [value[6], value[15]], 'o', color=clrs[colornum + len(experimentalStatsPerMouse.keys())])
                    colornum+=1
                else:
                    plt.plot([5,8], [value[6], value[15]], 'o', color='black')                                
            plt.legend()    
            #plt.ylim(0,15)   
            plt.xticks([1.5, 4.5, 7.5], ['24', 'Light', 'Dark'], rotation=60)
            plt.ylabel('Frequency (1/h)')
            plt.xlabel('Condition')
            plt.title('REM Frequencies')
            
            #######WAKE FREQUENCY########
            plt.figure()
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlWAKEFrequency), controlLightWAKEFrequency, controlDarkWAKEFrequency], label='Control')   
            barlistExperimental=plt.bar([2,5,8], height = [np.mean(experimentalWAKEFrequency), experimentalLightWAKEFrequency, experimentalDarkWAKEFrequency], label='Experimental')
            #color bars, comes after bars but before legend, legend then comes before point plot to keep them out of it
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')
            barlistExperimental[0].set_color('b')
            barlistExperimental[1].set_color('b')
            barlistExperimental[2].set_color('b')
            #Plot individual mouse points for full recording control bar            
            for i in range(len(controlWAKEFrequency)):
                if miceLabels:
                    plt.plot([1], controlWAKEFrequency[i], 'o', color=clrs[i])
                else:                   
                    plt.plot([1], controlWAKEFrequency[i], 'o', color='black')
            #Plot individual mouse points for full recording experimental bar     
            for i in range(len(experimentalWAKEFrequency)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalWAKEFrequency[i], 'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalWAKEFrequency[i], 'o', color=clrs[i+ len(experimentalStatsPerMouse.keys())])
                else:                   
                    plt.plot([2], experimentalWAKEFrequency[i], 'o', color='black')
                
            #Print individual mouse points in control columns for light and dark bars
            colornum=0
            for key, value in controlStatsPerMouseAverages.items():
                if miceLabels:
                    plt.plot([4,7], [value[7], value[16]], 'o', label=key, color=clrs[colornum])
                    colornum+=1
                else:
                    plt.plot([4,7], [value[7], value[16]], 'o', color='black')
                    
            #Print individual mouse points in experimental columns for light and dark bars
            colornum=0
            for key, value in experimentalStatsPerMouseAverages.items():
                if miceLabels:
                    if experimentalMice[colornum] == controlMice[colornum]:
                        plt.plot([5,8], [value[7], value[16]], 'o', color=clrs[colornum])
                    else:
                        plt.plot([5,8], [value[7], value[16]], 'o', label=key, color=clrs[colornum + len(experimentalStatsPerMouse.keys())])
                    colornum+=1
                else:
                    plt.plot([5,8], [value[7], value[16]], 'o', color='black')                                
            plt.legend()    
            #plt.ylim(0,15)   
            plt.xticks([1.5, 4.5, 7.5], ['24', 'Light', 'Dark'], rotation=60)
            plt.ylabel('Frequency (1/h)')
            plt.xlabel('Condition')
            plt.title('WAKE Frequencies')
            
            #######NREM FREQUENCY########
            plt.figure()
            barlistControl=plt.bar([1,4,7], height = [np.mean(controlNREMFrequency), controlLightNREMFrequency, controlDarkNREMFrequency], label='Control')   
            barlistExperimental=plt.bar([2,5,8], height = [np.mean(experimentalNREMFrequency), experimentalLightNREMFrequency, experimentalDarkNREMFrequency], label='Experimental')
            #color bars, comes after bars but before legend, legend then comes before point plot to keep them out of it
            barlistControl[0].set_color('gray')
            barlistControl[1].set_color('gray')
            barlistControl[2].set_color('gray')
            barlistExperimental[0].set_color('b')
            barlistExperimental[1].set_color('b')
            barlistExperimental[2].set_color('b')
            #Plot individual mouse points for full recording control bar            
            for i in range(len(controlNREMFrequency)):
                if miceLabels:
                    plt.plot([1], controlNREMFrequency[i], 'o', color=clrs[i])
                else:                   
                    plt.plot([1], controlNREMFrequency[i], 'o', color='black')
            #Plot individual mouse points for full recording experimental bar     
            for i in range(len(experimentalNREMFrequency)):
                if miceLabels:
                    if experimentalMice[i] == controlMice[i]:
                        plt.plot([2], experimentalNREMFrequency[i], 'o', color=clrs[i])
                    else:
                        plt.plot([2], experimentalNREMFrequency[i], 'o', color=clrs[i+ len(experimentalStatsPerMouse.keys())])
                else:                   
                    plt.plot([2], experimentalNREMFrequency[i], 'o', color='black')
                
            #Print individual mouse points in control columns for light and dark bars
            colornum=0
            for key, value in controlStatsPerMouseAverages.items():
                if miceLabels:
                    plt.plot([4,7], [value[8], value[17]], 'o', label=key, color=clrs[colornum])
                    colornum+=1
                else:
                    plt.plot([4,7], [value[8], value[17]], 'o', color='black')
                    
            #Print individual mouse points in experimental columns for light and dark bars
            colornum=0
            for key, value in experimentalStatsPerMouseAverages.items():
                if miceLabels:
                    if experimentalMice[colornum] == controlMice[colornum]:
                        plt.plot([5,8], [value[8], value[17]], 'o', color=clrs[colornum])
                    else:
                        plt.plot([5,8], [value[8], value[17]], 'o', label=key, color=clrs[colornum + len(experimentalStatsPerMouse.keys())])
                    colornum+=1
                else:
                    plt.plot([5,8], [value[8], value[17]], 'o', color='black')                                
            plt.legend()    
            #plt.ylim(0,15)   
            plt.xticks([1.5, 4.5, 7.5], ['24', 'Light', 'Dark'], rotation=60)
            plt.ylabel('Frequency (1/h)')
            plt.xlabel('Condition')
            plt.title('NREM Frequencies')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#Modified version of sleep stats to account for light / dark cycles
def sleep_stats(ppath, recordings, ma_thr=10.0, tstart=0, tend=-1, pplot=True, csv_file=''):
    """
    Calculate average percentage of each brain state,
    average duration and average frequency
    plot histograms for REM, NREM, and Wake durations
    @PARAMETERS:
    ppath      -   base folder
    recordings -   single string specifying recording or list of recordings

    @OPTIONAL:
    ma_thr     -   threshold for wake periods to be considered as microarousals
    tstart     -   only consider recorded data starting from time tstart, default 0s
    tend       -   only consider data recorded up to tend s, default -1, i.e. everything till the end
    pplot      -   generate plot in the end; True or False
    csv_file   -   file where data should be saved as csv file (e.g. csv_file = '/home/Users/Franz/Documents/my_data.csv')

    @RETURN:
        ndarray of percentages (# mice x [REM,Wake,NREM])
        ndarray of state durations
        ndarray of transition frequency / hour
    """
    if type(recordings) != list:
        recordings = [recordings]

    Percentage = {}
    Duration = {}
    Frequency = {}
    mice = []
    for rec in recordings:
        idf = re.split('_', os.path.split(rec)[-1])[0]
        if not idf in mice:
            mice.append(idf)
        Percentage[idf] = {1:[], 2:[], 3:[]}
        Duration[idf] = {1:[], 2:[], 3:[]}
        Frequency[idf] = {1:[], 2:[], 3:[]}
    nmice = len(Frequency)    

    for rec in recordings:
        #idf is the mouse ID
        idf = re.split('_', os.path.split(rec)[-1])[0]
        SR = sleepy.get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        dt = NBIN * 1/SR

        # load brain state
        M, K = sleepy.load_stateidx(ppath, rec)
        kcut = np.where(K >= 0)[0]
        M = M[kcut]
        
        # polish out microarousals
        M[np.where(M==5)] = 2                
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if len(s)*dt <= ma_thr:
                M[s] = 3
        
        ########### CALCULATE ISTART AND IEND ################
        if type(tstart) == int: #If we pass in an integer, proceed as normal
            istart = int(np.round((1.0 * tstart) / dt))
            if tend==-1:
                iend = len(M)-1
            else:
                iend = int(np.round((1.0*tend) / dt))
    
            midx = np.arange(istart,iend+1) #Generate integers for the range from istart to iend, +1 because iarange isn't inclusive of the last value so we force it
            Mcut = M[midx] #Cut M, which is the actual full recording, with the range we specify with istart and iend
            nm = len(Mcut)*1.0 #Get the length of that range
            
                # get percentage of each state
            for s in [1,2,3]:
                Percentage[idf][s].append(len(np.where(Mcut==s)[0]) / nm)
                
            # get frequency of each state
            for s in [1,2,3]:
                Frequency[idf][s].append( len(sleepy.get_sequences(np.where(Mcut==s)[0])) * (3600. / (nm*dt)) )
                
            # get average duration for each state
            for s in [1,2,3]:
                seq = sleepy.get_sequences(np.where(Mcut==s)[0])
                Duration[idf][s] += [len(i)*dt for i in seq] 
                
#if we pass in a list of start times and a list of end times in the case of multiple light periods for instance, account for multiple periods to run stats on                        
        else: 
            istart = []
            iend = []
            for eachValue in tstart: #create istart and iend values from tstart and tend values
                istartTemp = int(np.round((1.0 * eachValue) / dt))
                istart.append(istartTemp)
            for eachValue in tend:
                iendTemp = int(np.round((1.0*eachValue) / dt))
                iend.append(iendTemp)
            
            #generate list of cuts
            #method 1
            midx = []
            for i in range(len(istart)):
#                midx.append(np.arange(istart[i],iend[i]+1)) #for each value in istart (and implicitly iend), generate the range of numbers between istart and iend for each pair of them 
                midx.append(np.arange(istart[i],iend[i]))
                
            Mcut = [] 
            nm = 0
            for eachRange in midx:
                McutSub = M[eachRange]
                Mcut.append(McutSub)
                nm += len(eachRange) #accumulate lenghts of full range
            #Mcut is now a list of ranges in M (which is the full recording range)
            
             # get percentage of each state
            for s in [1,2,3]:
                numState = 0
                for eachRange in Mcut:
                    eachRangeNumState = len(np.where(eachRange==s)[0])
                    numState += eachRangeNumState
                Percentage[idf][s].append(numState / nm)
            
            # get frequency of each state
            for s in [1,2,3]:
                numSequences = 0
                for eachRange in Mcut:
                    eachRangeNumSequences = len(sleepy.get_sequences(np.where(eachRange==s)[0]))
                    numSequences += eachRangeNumSequences                
                Frequency[idf][s].append( numSequences * (3600. / (nm*dt)) )

            
             # get average duration for each state
            for s in [1,2,3]:
                for eachRange in Mcut: #for the first light period and the 2nd light period
                    eachRangeSeq = sleepy.get_sequences(np.where(eachRange==s)[0])
                    Duration[idf][s] += [len(i)*dt for i in eachRangeSeq]
                    
    #proceed as normal    
    PercMx = np.zeros((nmice,3))
    i=0
    for k in mice:
        for s in [1,2,3]:
            PercMx[i,s-1] = np.array(Percentage[k][s]).mean()
        i += 1
    PercMx *= 100
        
    FreqMx = np.zeros((nmice,3))
    i = 0
    for k in mice:
        for s in [1,2,3]:
            FreqMx[i,s-1] = np.array(Frequency[k][s]).mean()
        i += 1
    
    DurMx = np.zeros((nmice,3))
    i = 0
    for k in mice:
        for s in [1,2,3]:
            DurMx[i,s-1] = np.array(Duration[k][s]).mean()
        i += 1
        
    DurHist = {1:[], 2:[], 3:[]}
    for s in [1,2,3]:
        DurHist[s] = np.squeeze(np.array(reduce(lambda x,y: x+y, [Duration[k][s] for k in Duration])))

    if pplot:
        clrs = sns.color_palette("husl", nmice)
        plt.ion()
        # plot bars summarizing results - Figure 1
        plt.figure(figsize=(10, 5))
        ax = plt.axes([0.1, 0.15, 0.2, 0.8])
        #plot bar for mean of mice
        plt.bar([1,2,3], PercMx.mean(axis=0), align='center', color='gray', fill=False)
        plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'], rotation=60)
        #plot individual mouse points
        for i in range(nmice):
            plt.plot([1,2,3], PercMx[i,:], 'o', label=mice[i], color=clrs[i])
        plt.ylabel('Percentage (%)')
        plt.legend(fontsize=9)
        plt.xlim([0.2, 3.8])
        sleepy.box_off(ax)
            
        ax = plt.axes([0.4, 0.15, 0.2, 0.8])
        plt.bar([1,2,3], DurMx.mean(axis=0), align='center', color='gray', fill=False)
        plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'], rotation=60)
        for i in range(nmice):
            plt.plot([1, 2, 3], DurMx[i, :], 'o', label=mice[i], color=clrs[i])
        plt.ylabel('Duration (s)')
        plt.xlim([0.2, 3.8])
        sleepy.box_off(ax)
            
        ax = plt.axes([0.7, 0.15, 0.2, 0.8])
        plt.bar([1,2,3], FreqMx.mean(axis=0), align='center', color='gray', fill=False)
        plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'], rotation=60)
        for i in range(nmice):
            plt.plot([1, 2, 3], FreqMx[i, :], 'o', label=mice[i], color=clrs[i])
        plt.ylabel('Frequency (1/h)')
        plt.xlim([0.2, 3.8])
        sleepy.box_off(ax)
        plt.show(block=False)    

        # plot histograms - Figure 2            
        plt.figure(figsize=(5, 10))
        ax = plt.axes([0.2,0.1, 0.7, 0.2])
#        h, edges = np.histogram(DurHist[1], bins=40, range=(0, 300), normed=1)
        h, edges = np.histogram(DurHist[1], bins=40, range=(0, 300))
        binWidth = edges[1] - edges[0]
        plt.bar(edges[0:-1], h*binWidth, width=5)
        plt.xlim((edges[0], edges[-1]))
        plt.xlabel('Duration (s)')
        plt.ylabel('Freq. REM')
        sleepy.box_off(ax)
        
        ax = plt.axes([0.2,0.4, 0.7, 0.2])
#        h, edges = np.histogram(DurHist[2], bins=40, range=(0, 1200), normed=1)
        h, edges = np.histogram(DurHist[2], bins=40, range=(0, 1200))
        binWidth = edges[1] - edges[0]
        plt.bar(edges[0:-1], h*binWidth, width=20)
        plt.xlim((edges[0], edges[-1]))
        plt.xlabel('Duration (s)')
        plt.ylabel('Freq. Wake')
        sleepy.box_off(ax)
        
        ax = plt.axes([0.2,0.7, 0.7, 0.2])
#        h, edges = np.histogram(DurHist[3], bins=40, range=(0, 1200), normed=1)
        h, edges = np.histogram(DurHist[3], bins=40, range=(0, 1200))
        binWidth = edges[1] - edges[0]
        plt.bar(edges[0:-1], h*binWidth, width=20)
        plt.xlim((edges[0], edges[-1]))
        plt.xlabel('Duration (s)')
        plt.ylabel('Freq. NREM')
        sleepy.box_off(ax)
        plt.show()

    if len(csv_file) > 0:
        mouse_list = [[m]*3 for m in mice]
        mouse_list = sum(mouse_list, [])
        state_list = ['REM', 'Wake', 'NREM']*nmice
        df = pd.DataFrame({'mouse':mouse_list, 'state':state_list, 'Perc':PercMx.flatten(), 'Dur':DurMx.flatten(), 'Freq':FreqMx.flatten()})
        df.to_csv(csv_file)

    return PercMx, DurMx, FreqMx, mice

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

def get_cycles(ppath, name):
    """
    extract the time points where dark/light periods start and end
    """
    act_dur = sleepy.get_infoparam(os.path.join(ppath, name, 'info.txt'), 'actual_duration')

    time_param = sleepy.get_infoparam(os.path.join(ppath, name, 'info.txt'), 'time')
    if len(time_param) == 0 or len(act_dur) == 0:
        return {'light': [(0,0)], 'dark': [(0,0)]}
    
    hour, mi, sec = [int(i) for i in re.split(':', time_param[0])]
    #a = sleepy.get_infoparam(os.path.join(ppath, name, 'info.txt'), 'actual_duration')[0]
    a,b,c = [int(i[0:-1]) for i in re.split(':', act_dur[0])]
    total_dur = a*3600 + b*60 + c
    
    # number of light/dark switches
    start_time = hour*3600 + mi*60 + sec
    end_time = start_time + total_dur
    seven_am = 7*3600
    seven_pm = 19*3600
    seven_am2 = 31*3600
    # start dark (early)
    if (start_time < seven_am):
        if end_time < seven_am:
            nswitch=0
        else:
            remainder = end_time - seven_am
            nswitch = 1 + int(np.floor(remainder)/(12*3600))
    # start light
    elif (start_time >= seven_am) and (start_time < seven_pm):
        if end_time < seven_pm:
            nswitch = 0
        else:
            remainder = end_time - seven_pm
            nswitch = 1 + int(np.floor(remainder)/(12*3600))
    else:
        if end_time < seven_am2:
            nswitch=0
        else:
            remainder = end_time - seven_am2
            nswitch = 1 + int(np.floor(remainder)/(12*3600)) 
    
    
    startPhase = ''
    switch_points = [0]
    cycle = {'light': [], 'dark':[]}
    
    if hour >= 7 and hour < 19:
        startPhase = 'light'
        # recording starts during light cycle
        a = 19*3600 - (hour*3600+mi*60+sec)
        for j in range(nswitch):
            switch_points.append(a+j*12*3600)
        for j in range(1, nswitch, 2):
            cycle['dark'].append(switch_points[j:j+2])
        for j in range(0, nswitch, 2):
            cycle['light'].append(switch_points[j:j+2])
        
    else:
        startPhase = 'dark'
        # recording starts during dark cycle
        a = 0
        if hour < 24:
            a = 24 - (hour*3600+mi*60+sec) + 7*3600
        else:
            a = 7*3600 - (hour*3600+mi*60+sec)
        for j in range(nswitch):
            switch_points.append(a+j*12*3600)
        for j in range(0, nswitch, 2):
            cycle['dark'].append(switch_points[j:j+2])
        for j in range(1, nswitch, 2):
            cycle['light'].append(switch_points[j:j+2])
        
    return cycle, total_dur, startPhase



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    

def sleep_ranksums(ppath, controlRecordings=[], experimentalRecordings=[], lightDark = False, noGraphs=True): #set pplot to true if you're running this apart from graph_conditions and want the original graphs generated by sleep_stats   
   ################# C O N T R O L ###########################
    
   if lightDark == True:
       #take in time arguments from get_cycle to more precisely process light and dark (as opposed to taking stricly the first and last half of the recording as before)
       #tstart and tend arguments will overwrite the default 0 value
      
       for rec in controlRecordings: #sleep_half_stats takes in 1 recording at a time and handles the tstarts and tends
           controlLightStats, controlDarkStats, mouseID = sleep_half_stats(ppath, rec)    
           
       for rec in experimentalRecordings:
           experimentalLightStats, exerimentalDarkStats, mouseID = sleep_half_stats(ppath, rec)
   
   else: 
       #assume you're not comparing light and dark phases
       controlPercents, controlDurations, controlFrequencies, controlMice = sleep_stats(ppath, controlRecordings, 10.0, 0, -1, pplot=False)
       experimentalPercents, experimentalDurations, experimentalFrequencies, experimentalMice = sleep_stats(ppath, experimentalRecordings, 10.0, 0, -1, pplot=False)

   #regroup REM NREM and WAKE stats into new arrays for first half
   
   controlStats = [controlPercents, controlDurations, controlFrequencies]
   experimentalStats = [experimentalPercents, experimentalDurations, experimentalFrequencies]
  
   #declare and regroup
   controlREMPercent = []  
   controlWAKEPercent = []
   controlNREMPercent = []

   for eachMouse in controlPercents:
       controlREMPercent.append(eachMouse[0])         
       controlWAKEPercent.append(eachMouse[1])
       controlNREMPercent.append(eachMouse[2])

         
   #declare and regroup
   controlREMDuration = []
   controlWAKEDuration = []
   controlNREMDuration = []
   
   for eachMouse in controlDurations:
       controlREMDuration.append(eachMouse[0]) 
       controlWAKEDuration.append(eachMouse[1])
       controlNREMDuration.append(eachMouse[2]) 
   
 
   #declare and regroup
   controlREMFrequency = []
   controlWAKEFrequency = []
   controlNREMFrequency = []  
   for eachMouse in controlFrequencies:
       controlREMFrequency.append(eachMouse[0]) 
       controlWAKEFrequency.append(eachMouse[1])
       controlNREMFrequency.append(eachMouse[2])  
    
    
   ######EXPERIMENTAL#############
   #regroup REM NREM and WAKE stats into new arrays for first half
   
   #declare and regroup
   experimentalREMPercent = []
   experimentalWAKEPercent = []
   experimentalNREMPercent = []
   
   for eachMouse in experimentalPercents:
       experimentalREMPercent.append(eachMouse[0]) 
       experimentalWAKEPercent.append(eachMouse[1])
       experimentalNREMPercent.append(eachMouse[2]) 

   #declare and regroup
   experimentalREMDuration = []
   experimentalWAKEDuration = []
   experimentalNREMDuration = []
   
   for eachMouse in experimentalDurations:
       experimentalREMDuration.append(eachMouse[0]) 
       experimentalWAKEDuration.append(eachMouse[1])
       experimentalNREMDuration.append(eachMouse[2]) 
         
    
   #declare and regroup
   experimentalREMFrequency = []
   experimentalWAKEFrequency = []
   experimentalNREMFrequency = []  
   for eachMouse in experimentalFrequencies:
       experimentalREMFrequency.append(eachMouse[0])
       experimentalWAKEFrequency.append(eachMouse[1])
       experimentalNREMFrequency.append(eachMouse[2]) 
              
       
       
   #Collect full stats across mice (not per mouse)
   controlStatsFull = []
   controlStatsFull.append(controlREMPercent)
   controlStatsFull.append(controlWAKEPercent)
   controlStatsFull.append(controlNREMPercent)
   controlStatsFull.append(controlREMDuration)
   controlStatsFull.append(controlWAKEDuration)
   controlStatsFull.append(controlNREMDuration)
   controlStatsFull.append(controlREMFrequency)
   controlStatsFull.append(controlWAKEFrequency)
   controlStatsFull.append(controlNREMFrequency)
   
   experimentalStatsFull = []
   experimentalStatsFull.append(experimentalREMPercent)
   experimentalStatsFull.append(experimentalWAKEPercent)
   experimentalStatsFull.append(experimentalNREMPercent)
   experimentalStatsFull.append(experimentalREMDuration)
   experimentalStatsFull.append(experimentalWAKEDuration)
   experimentalStatsFull.append(experimentalNREMDuration)
   experimentalStatsFull.append(experimentalREMFrequency)
   experimentalStatsFull.append(experimentalWAKEFrequency)
   experimentalStatsFull.append(experimentalNREMFrequency)
          

#might have to move to plotting function in case of comparing light and dark phases
    
   #Compute p-values for first half (control vs experimental)
   percentREM = stats.ranksums(controlREMPercent, experimentalREMPercent)[1]
   percentNREM = stats.ranksums(controlNREMPercent, experimentalNREMPercent)[1]
   percentWAKE = stats.ranksums(controlWAKEPercent, experimentalWAKEPercent)[1]
      
   durationREM = stats.ranksums(controlREMDuration, experimentalREMDuration)[1]
   durationNREM = stats.ranksums(controlNREMDuration, experimentalNREMDuration)[1]
   durationWAKE =  stats.ranksums(controlWAKEDuration, experimentalWAKEDuration)[1]
    
   frequencyREM = stats.ranksums(controlREMFrequency, experimentalREMFrequency)[1]
   frequencyNREM = stats.ranksums(controlNREMFrequency, experimentalNREMFrequency)[1]
   frequencyWAKE = stats.ranksums(controlWAKEFrequency, experimentalWAKEFrequency)[1]
    
    
   pValuesPercent = (percentREM, percentWAKE, percentNREM)
   pValuesDuration = (durationREM, durationWAKE, durationNREM)
   pValuesFrequency = (frequencyREM, frequencyWAKE,frequencyNREM)
    
   print("p-values for full-recording percentages are")
   print(pValuesPercent)
    
   print("p-values for full-recording durations are")
   print(pValuesDuration)
    
   print("p-values for full-recording frequencies are")
   print(pValuesFrequency)
      

      
   #vestigial code
   if noGraphs:
       return controlStatsFull, experimentalStatsFull
   else:      
       return controlStats, experimentalStats, controlMice, experimentalMice
 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#Designed to only handle 1 recording at a time, strictly a helper function for main plot function
def sleep_half_stats(ppath, recording):  
    
    ##### LIGHT STATS ####       
    lightDarkCycle, total_dur, startPhase = get_cycles(ppath,recording)   
    
    darktstart = (lightDarkCycle.get("dark"))[0][0]
    darktend = (lightDarkCycle.get("dark"))[0][1]
    
    #handles the first light period
    lighttstart1 = (lightDarkCycle.get("light"))[0][0]
    lighttend1 = (lightDarkCycle.get("light"))[0][1]
    #handles the second light period
    lighttstart2 = darktend 
    lighttend2 = total_dur
        
    lighttstart = [lighttstart1, lighttstart2]
    lighttend = [lighttend1, lighttend2]
    
    
    #get array data for first half of recordings and break it up into percents, durations and frequencies -- takes in a list of start and end points
    #sleep_stats handles this case appropriately
    lightPercents, lightDurations, lightFrequencies, mouseIDfromStats1 = sleep_stats(ppath, recording, 10.0, tstart=lighttstart, tend=lighttend, pplot = False)

    #declare and regroup
    lightREMPercent = lightPercents[0][0]
    lightWAKEPercent = lightPercents[0][1]
    lightNREMPercent = lightPercents[0][2]

    #declare and regroup
    lightREMDuration = lightDurations[0][0]
    lightWAKEDuration = lightDurations[0][1]
    lightNREMDuration = lightDurations[0][2] 
 
    #declare and regroup
    lightREMFrequency = lightFrequencies[0][0]
    lightWAKEFrequency = lightFrequencies[0][1]
    lightNREMFrequency = lightFrequencies[0][2]  
    
    ####### Dark Phase #######
    #get array data for second half of recordings and break it up into percents, durations and frequencies
    darkPercents, darkDurations, darkFrequencies, mouseIDfromStats2 = sleep_stats(ppath, recording, 10.0, tstart=darktstart, tend=darktend, pplot = False)
    
    #regroup REM NREM and WAKE stats into new arrays for second half
    darkREMPercent = darkPercents[0][0]
    darkWAKEPercent = darkPercents[0][1]
    darkNREMPercent = darkPercents[0][2] 
    
    #declare and regroup
    darkREMDuration = darkDurations[0][0]
    darkWAKEDuration = darkDurations[0][1]
    darkNREMDuration = darkDurations[0][2] 
    
    #declare and regroup
    darkREMFrequency = darkFrequencies[0][0]
    darkWAKEFrequency = darkFrequencies[0][1]
    darkNREMFrequency = darkFrequencies[0][2]  
            
    lightStatsReturn = []
    lightStatsReturn.append(lightREMPercent)
    lightStatsReturn.append(lightWAKEPercent)
    lightStatsReturn.append(lightNREMPercent)
    lightStatsReturn.append(lightREMDuration)
    lightStatsReturn.append(lightWAKEDuration)
    lightStatsReturn.append(lightNREMDuration)
    lightStatsReturn.append(lightREMFrequency)
    lightStatsReturn.append(lightWAKEFrequency)
    lightStatsReturn.append(lightNREMFrequency)
    
    darkStatsReturn = []
    darkStatsReturn.append(darkREMPercent)
    darkStatsReturn.append(darkWAKEPercent)
    darkStatsReturn.append(darkNREMPercent)
    darkStatsReturn.append(darkREMDuration)
    darkStatsReturn.append(darkWAKEDuration)
    darkStatsReturn.append(darkNREMDuration)
    darkStatsReturn.append(darkREMFrequency)
    darkStatsReturn.append(darkWAKEFrequency)
    darkStatsReturn.append(darkNREMFrequency)
    
    #get mouseID and return it so we can start grouping the stats together per mouse in dictionaries
    mouseID = re.split('_', os.path.split(recording)[-1])[0]
    
    return lightStatsReturn, darkStatsReturn, mouseID



