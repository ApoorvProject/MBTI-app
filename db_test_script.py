import sqlite3
import os

conn = sqlite3.connect('personality_data.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM livedata_db WHERE date"\
          " BETWEEN '2022-01-01 00:00:00' AND DATETIME('now')")
results = c.fetchall()
conn.close()
print(results)