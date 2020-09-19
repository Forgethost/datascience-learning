
from geopy.geocoders import Nominatim

def myGeoCoderLatitude(x):
    city = x.split(',')[0]
    locator = Nominatim(user_agent='myGeocoder')
    location = locator.geocode('{},CT'.format(city))
    latitude = location.latitude
    return latitude
    

def myGeoCoderLongitude(x):
    city = x.split(',')[0]
    locator = Nominatim(user_agent='myGeocoder')
    location = locator.geocode('{},CT'.format(city))
    longitude = location.longitude
    return longitude