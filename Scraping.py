import glassdoor_scraper as gs
import pandas as pd

path = "chromedriver.exe"

df = gs.get_jobs("Data scientist", 15, False, path, 15)

#pd.to_csv(df)