from gmplot import gmplot
from pymongo import MongoClient


def create_map():

    client = MongoClient('localhost', 27017)
    db = client.Yelp

    locations = []
    for x in db.restaurants.find({},{"stars": 1, "latitude": 1, "longitude": 1}):
        locations.append((x["latitude"], x["longitude"], x["stars"]))

    gmap = gmplot.GoogleMapPlotter.from_geocode("United States")

    for i in range(1,len(locations)):
        if locations[i][2] < 2:
            gmap.marker(locations[i][0], locations[i][1], 'red')
        elif locations[i][2] < 3:
            gmap.marker(locations[i][0], locations[i][1], 'orange')
        elif locations[i][2] < 4:
            gmap.marker(locations[i][0], locations[i][1], 'yellow')
        else:
            gmap.marker(locations[i][0], locations[i][1], 'green')

    gmap.draw("../data/map.html")


if __name__ == '__main__':
    create_map()
