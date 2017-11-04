import os
import json
import csv


files = [os.path.join('data', file) for file in os.listdir('data') \
            if file.endswith('.json')]

with open(os.ppath.join('data', 'monuments.txt'), 'w', newline='') as fout:
    writer = csv.writer(fout, delimiter=',')
    writer.writerow(['place', 'state', 'lat', 'lon', 'count'])
    for file in files:
        with open(file, 'r') as f:
            coordinates = json.load(f)
            for key in coordinates:
                place, state = key.split(', AT')[0].split(', ')
                lat = coordinates[key]['lat']
                lon = coordinates[key]['lon']
                count = coordinates[key]['count']
                writer.writerow([place, state, lat, lon, count])
