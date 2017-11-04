import os
import geocoder
import csv
import numpy as np
import time
import json


folder = os.path.join('data', 'monuments')
files = [file for file in os.listdir(folder) if file.endswith('.csv')]

for file in files:
    coordinates = {}
    state = file.split('_')[0]
    filename = file.split('.')[0]

    with open(os.path.join(folder, file), 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        data = np.array(list(reader))

        for place in data[:, 1]:
            time.sleep(0.05)
            address = '{}, {}, AT'.format(place, state)
            if address not in coordinates:
                print(address)
                g = geocoder.arcgis(address)

                trials = 10
                for i in range(trials):
                    print(g.status)
                    if g.status == 'OK':
                        entry = {
                                'lat':g.latlng[0],
                                'lon':g.latlng[1],
                                'bbox': g.bbox,
                                'count': 1}
                        print(entry)
                        coordinates[address] = entry
                        break
                    elif g.status == 'ZERO_RESULTS':
                        g = geocoder.arcgis(address)
                        if i == trials - 1:
                            print("ERROR")
                            coordinates[address] = g.status
                    else:
                        print('ERROR')
                        print(g.current_result)
                        time.sleep(1)
                        g = geocoder.arcgis(address)

                location = results[0]['geometry']['location']
                print(location['lat'], location['lng'])
            else:
                coordinates[address]['count'] += 1

    outputfile = os.path.join('data', filename+".json")
    with open(outputfile, 'w') as f:
        json.dump(coordinates, f, sort_keys=True, indent=4)
