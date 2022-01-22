import requests
import csv
import os
from pymongo import MongoClient, collection
import json
import sys

def map_city(mappings, city):
    if city in mappings:
        return mappings[city]
    if "kraj" in city.lower():
        return city
    if "republika" in city:
        return "Česká republika"

# constants
DB = None
COLLECTIONS = None
URLS = None 
ENCODINGS = None

def main():
    # load config
    with open(sys.path[0] + "/config.json") as config:
        conf_json = json.load(config)
        connection_string = conf_json["connection_string"]
        client = MongoClient(connection_string)
        DB = client["UPA"]
        COLLECTIONS = conf_json["collections"]
        URLS = conf_json["urls"]
        ENCODINGS = conf_json["encodings"]

    # download datasets
    for url, collect in zip(URLS, COLLECTIONS):
        if not os.path.isfile(f"{collect}.csv"):
            r = requests.get(url, allow_redirects=True)
            open(f"{collect}.csv", 'wb').write(r.content)

    # drop DB
    for collect in COLLECTIONS:
        DB[collect].drop()

    city_mapping = {}
    # prepare data from csv and insert into mongoDB
    for url, collect, encoding in zip(URLS, COLLECTIONS, ENCODINGS):
        with open(f"{collect}.csv", encoding=encoding, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';' if collect == "medical_facilities" else ',')

            keys = next(csv_reader)
            collection_n = DB[collect]
            records = list()

            for line in csv_reader:
                zip_iter = zip(keys, line)
                single_record = dict(zip_iter)
                records.append(single_record)

            if collect == "medical_facilities":
                for record in records:
                    city_mapping[record["Okres"]] = record["Kraj"]
            else:
                city_regions = [map_city(city_mapping, record["vuzemi_txt"])
                                for record in records]
                for record, region in zip(records, city_regions):
                    record["Kraj"] = region

            collection_n.insert_many(records)

if __name__ == '__main__':
    main()
