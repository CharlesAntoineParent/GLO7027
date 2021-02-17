import datetime
import os
import utils
from tqdm import tqdm
from matplotlib import pyplot
import pandas as pd

journaux = [folder for folder in os.listdir('data/analytics') if not folder.startswith('.')]

startDate = datetime.date(2019,1,1)
endDate = datetime.date(2019,7,31)
dateList = [startDate + datetime.timedelta(days=day) for day in range( (endDate - startDate).days + 1 )]

journalPopularityByday = dict()


for journal in journaux:
    print(journal)
    journalPopularityByday[journal] = dict()
    pbar = tqdm(total=100)
    for date in dateList:
        journalPopularityByday[journal][str(date)] = sum(utils.dayPopularity(str(date),journal).values())
        pbar.update(100 / len(dateList))
    pbar.close()


series = pd.DataFrame(journalPopularityByday)

series.plot()
pyplot.show()

