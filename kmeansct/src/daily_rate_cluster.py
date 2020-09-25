from io import BytesIO
from PyPDF2 import PdfFileReader
import requests
import re
import pandas as pd
import os
import datetime
from datetime import datetime as dt
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import numpy as np


raw_data_path = os.path.join(os.path.pardir,"data","raw","CaseCountData_texas.csv")
zip_data_path = os.path.join(os.path.pardir,"data","raw","zipMap.csv")
raw_data_path_daily_rate = os.path.join(os.path.pardir,"data","raw","case_counts_daily_rate.csv")
predictions_data_band = os.path.join(os.path.pardir,"data","processed","predictions","predictions_bands.csv")
centers_daily_path = os.path.join(os.path.pardir,"data","processed","predictions","centers_daily.csv")
predictions_data_path_zip = os.path.join(os.path.pardir,"data","processed","predictions","predictions_daily_rate_zip.csv")
predictions_data_path = os.path.join(os.path.pardir,"data","processed","predictions","predictions_tx.csv")
predictions_data_path_daily_rate = os.path.join(os.path.pardir,"data","processed","predictions","predictions_daily_rate_rank.csv")
wcss_plt_path = os.path.join(os.path.pardir,"plots","daily_rate","wcss_plot.png")
elbow_plt_path = os.path.join(os.path.pardir,"plots","daily_rate","elbow_plot.png")
daily_scatter_plt_path = os.path.join(os.path.pardir,"plots","daily_rate","cluster_scatter.png")


def kmeans_model_predict(X_data, k):
    date = get_daily_rate_pdf_date()
    clf = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=10, random_state=0)
    kmeans = clf.fit(X_data)
    centers = pd.DataFrame(clf.cluster_centers_, columns=['centroids'])
    print(centers)
    Y_data = kmeans.predict(X_data)
    # return Y_data
    Y_df = pd.DataFrame(Y_data)
    Y_df.columns = ["group"]
    # print(Y_df)
    predicted_df = pd.concat([df.iloc[0:, 0:4], Y_df], axis=1)
    predicted_df.insert(0, "state", "CT")
    predicted_df_zip = predicted_df.merge(df_zip, how='left', left_on='city', right_on='City')
    Y_labels = Y_data.tolist()
    print(Y_data.shape)
    print("Y lables", len(Y_labels))
    # print(X_data[Y_data==2])
    print("x axis", Y_data[Y_data == 0])
    centers['group'] = np.arange(0, 6, 1)
    print(centers)
    centers['center_rank'] = centers["centroids"].rank(ascending=True)
    # centers['band'] = centers["centroids"].apply(assignBand)

    centers.sort_values(by=['center_rank'], ascending=True, inplace=True)
    print(centers)
    print("y axis", X_data[Y_data == 0]["case_counts"].tolist())
    print("Y axis annotate", X_data[Y_data == 0]["case_counts"].shape)
    centers_daily = centers.copy()
    centers_daily.insert(3, 'date', date)
    print(centers_daily)
    centers_daily.to_csv(centers_daily_path, index=None)
    # join zip df with centers and write data
    predicted_df_zip = predicted_df_zip.merge(centers, how='left', on='group')
    predicted_df_zip.to_csv(predictions_data_path_zip, index=None)

    # create bands df
    df_bands = predicted_df_zip.merge(centers, how='left', on='group')
    # print(df_bands.info())
    # print(predicted_df_zip.info())

    ###Write df daily data with zip
    df_bands.to_csv(predictions_data_band, index=None)

    #### Write daily final data
    predicted_df_daily = predicted_df.merge(centers, how='left', on='group')
    predicted_df_daily.to_csv(predictions_data_path_daily_rate, index=None)

    scatter_plot_cluster(X_data, Y_data)


def scatter_plot_cluster(X_data, Y_data):
    plt.figure(figsize=(12,8),dpi=100)
    plt.scatter(Y_data[Y_data==0],X_data[Y_data==0]["case_counts"].tolist(),s=50,c='gray',label='Cluster 1')
    plt.scatter(Y_data[Y_data==1],X_data[Y_data==1]["case_counts"].tolist(),s=50,c='red',label='Cluster 2')
    plt.scatter(Y_data[Y_data==2],X_data[Y_data==2]["case_counts"].tolist(),s=50,c='blue',label='Cluster 3')
    plt.scatter(Y_data[Y_data==3],X_data[Y_data==3]["case_counts"].tolist(),s=50,c='green',label='Cluster 4')
    plt.scatter(Y_data[Y_data==4],X_data[Y_data==4]["case_counts"].tolist(),s=50,c='yellow',label='Cluster 5')
    plt.scatter(Y_data[Y_data==5],X_data[Y_data==5]["case_counts"].tolist(),s=50,c='orange',label='Cluster 5')

    # plt.show()
    plt.savefig(daily_scatter_plt_path)

def wcss_plt(X_data):
    # WCSS within cluster sum of squares - sum of
    # the distances of observation from cluster centroids.
    # We will use this to find elbow point
    wcss = []

    for i in range(1, 11):
        clf = KMeans(n_clusters=i, init='k-means++', max_iter=1000, n_init=10)

        clf.fit(X_data)
        wcss.append(clf.inertia_)
    # print(len(wcss))
    print(wcss)
    plt.plot(range(1, 11), wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    # plt.show()
    plt.savefig(wcss_plt_path)


def elbow_plt(X_data):
    # Import ElbowVisualizer

    model = KMeans(init='k-means++', max_iter=1000, n_init=10, random_state=0)
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(3, 12), timings=False)
    visualizer.fit(X_data)  # Fit the data to the visualizer
    # visualizer.show()  # Finalize and render the figure
    print("Elbow Value by Yellowbricks ", visualizer.elbow_value_)
    plt.savefig(elbow_plt_path)


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


def myGeoCoderZipcode(x):
    city = x.split(',')[0]
    locator = Nominatim(user_agent='myGeocoder')
    location = locator.geocode('1 Main Street,{},CT'.format(city))
    zipcode = location.address[-2]
    return zipcode


def get_daily_rate_pdf_date():
    today_weekday = dt.weekday(dt.today())

    if today_weekday == 0:
        pdf_extract_day = -4
    elif today_weekday == 1:
        pdf_extract_day = -5
    elif today_weekday == 2:
        pdf_extract_day = -6
    elif today_weekday == 3:
        pdf_extract_day = -7
    elif today_weekday == 4:
        pdf_extract_day = -1
    elif today_weekday == 5:
        pdf_extract_day = -2
    elif today_weekday == 6:
        pdf_extract_day = -3

    pdf_date = dt.today() + datetime.timedelta(days=pdf_extract_day)
    pdf_date_str = dt.strftime(pdf_date, "%m%d%Y")[1:]

    return pdf_date_str

def get_cumulative_count_pdf_date():
    today_weekday = dt.weekday(dt.today())

    if (today_weekday > 0) and (today_weekday < 6):
        pdf_extract_day = -1
    elif today_weekday == 0:
        pdf_extract_day = -3
    elif today_weekday == 6:
        pdf_extract_day = -2
    else:
        raise Exception("Weekday out of range :{}".format(today_weekday))

    pdf_date = dt.today() + datetime.timedelta(days=pdf_extract_day)
    pdf_date_str = dt.strftime(pdf_date, "%m%d%Y")[1:]

    return pdf_date_str


def ct_cumulative_counts():
    data_list = []
    # today = datetime.strftime(datetime.today(),"%m%d%Y")[1:]
    today = get_cumulative_count_pdf_date()
    print(today)
    # url="https://portal.ct.gov/-/media/Coronavirus/CTDPHCOVID19summary9112020.pdf"
    url = "https://portal.ct.gov/-/media/Coronavirus/CTDPHCOVID19summary{}.pdf".format(today)
    print(url)
    ###########################################
    # uncomment  below lines for today's gove data
    pdf_data = requests.get(url)
    pdf_content = BytesIO(pdf_data.content)
    reader = PdfFileReader(pdf_content)
    ###################################
    # REmove below 2 lines for actual today's gov data"
    # pdffile=open(r"c:\users\babub\downloads\CTDPHCOVID19summary9112020.pdf","rb")
    # reader = PdfFileReader(pdffile)
    ###############################################
    for eachpage in range(4, 5):
        try:
            case_count_page = reader.getPage(eachpage).extractText()
            if len(re.findall("Cumulative Number of COVID", case_count_page)) > 0:
                case_counts = re.findall("Andover[\s\S]+Preston", case_count_page)[0]
                # city_counts = re.findall("\S+[\n ]+[\n]+\S+[\n ]+[\n]+\S+[\n ]+[\n]+",case_counts)
                city_counts = re.findall("\S+?\s*\S*[\n ]+[\n]+\S+[\n ]+[\n]+\S+[\n ]+[\n]+", case_counts)
                for each_city in city_counts:
                    try:
                        int(re.split("[\n ]+[\n]+", each_city)[1])
                        data_list.append(re.split("[\n ]+[\n]+", each_city)[0:3])
                    except:
                        pass

                df = pd.DataFrame(data_list, columns=["city", "case_counts", "probable_counts"])
                df.to_csv(raw_data_path, header=True, index=None)
        except Exception as e:
            print("Exception occurred", e)


def daily_ct_covid_rate():
    data_list = []
    # today = datetime.strftime(datetime.today(),"%m%d%Y")[1:]
    today = get_daily_rate_pdf_date()
    # print(today)
    # url="https://portal.ct.gov/-/media/Coronavirus/CTDPHCOVID19summary9172020.pdf"
    url = "https://portal.ct.gov/-/media/Coronavirus/CTDPHCOVID19summary{}.pdf".format(today)
    print(url)
    ###########################################
    # uncomment  below lines for today's gove data
    # pdf_data = requests.get(url)
    # pdf_content = BytesIO(pdf_data.content)
    # reader = PdfFileReader(pdf_content)
    ###################################
    # REmove below 2 lines for actual today's gov data"
    pdffile = open(r"c:\users\babub\downloads\CTDPHCOVID19summary9172020.pdf", "rb")
    reader = PdfFileReader(pdffile)
    ###############################################
    for eachpage in range(4, 5):
        try:
            case_count_page = reader.getPage(eachpage).extractText()
            if len(re.findall("Average Daily Rate of COVID", case_count_page)) > 0:
                # case_counts = re.findall("Andover[\s\S]+Woodstock",case_count_page)[0]
                case_counts = re.findall("Andover[\s\S]+", case_count_page)[0]
                # city_counts = re.findall("\S+[\n ]+[\n]+\S+[\n ]+[\n]+\S+[\n ]+[\n]+",case_counts)
                city_counts = re.findall("\S+?\s*\S*[\n ]+[\n]+\S+[\n ]+[\n]+\S+[\n ]+[\n]+\S+[\n ]+[\n]+", case_counts)
                for each_city in city_counts:
                    try:
                        int(re.split("[\n ]+[\n]+", each_city)[1])
                        data_list.append(re.split("[\n ]+[\n]+", each_city)[0:4])
                    except:
                        len_str = len(re.split("[\n ]+[\n]+", each_city))
                        if len_str == 6:
                            data_list.append(re.split("[\n ]+[\n]+", each_city)[1:-1])
                    finally:
                        pass

                df = pd.DataFrame(data_list, columns=["city", "population", "weekly_cases", "case_counts"])
                df.to_csv(raw_data_path_daily_rate, header=True, index=None)
        except Exception as e:
            print("Exception occurred", e)


if __name__ == '__main__':
    # daily rate only on Thursdays
    daily_ct_covid_rate()
    # create dataframe
    df = pd.read_csv(raw_data_path_daily_rate)
    df_zip = pd.read_csv(zip_data_path, dtype=object)
    X_data = df.loc[0:, ['case_counts']]
    # print(X_data.head())
    # Elbow value from yellow bricks
    elbow_plt(X_data)

    # set k to 6
    k=6
    kmeans_model_predict(X_data,k)




