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
from sklearn.preprocessing import StandardScaler



#import kmeansct

raw_data_path = os.path.join(os.path.pardir,"data","raw","CaseCountData_texas.csv")
zip_data_path = os.path.join(os.path.pardir,"data","raw","zipMap.csv")
raw_data_path_daily_rate = os.path.join(os.path.pardir,"data","raw","case_counts_daily_rate.csv")
predictions_data_band = os.path.join(os.path.pardir,"data","processed","predictions","predictions_bands.csv")
predictions_data_path_zip = os.path.join(os.path.pardir,"data","processed","predictions","predictions_daily_rate_zip.csv")
predictions_data_path = os.path.join(os.path.pardir,"data","processed","predictions","predictions_tx.csv")
predictions_data_path_cumulative = os.path.join(os.path.pardir,"data","processed","predictions","predictions_cumulative.csv")
predictions_data_path_daily_rate = os.path.join(os.path.pardir,"data","processed","predictions","predictions_daily_rate_1d.csv")
wcss_plt_path = os.path.join(os.path.pardir,"plots","cumulative_count","wcss_plot.png")
elbow_plt_path = os.path.join(os.path.pardir,"plots","cumulative_count","elbow_plot.png")
daily_scatter_plt_path = os.path.join(os.path.pardir,"plots","cumulative_count","cluster_scatter.png")



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


def kmeans_model_predict_cumulative(X_data, k):
    clf = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=10, random_state=0)
    kmeans = clf.fit(X_data)
    centers = pd.DataFrame(clf.cluster_centers_, columns=['centroids'])
    print(centers)
    Y_data = kmeans.predict(X_data)
    # rank centroids df
    centers['group_rank'] = centers['centroids'].rank(ascending=True)
    centers['group'] = pd.Series(centers.index.tolist())
    print(centers)
    # predictions_data_path
    Y_df = pd.DataFrame(Y_data)
    Y_df.columns = ["group"]
    # print(Y_df)
    predicted_df = pd.concat([df.iloc[0:, 0:3], Y_df], axis=1)
    predicted_df = predicted_df.merge(centers, how='inner', left_on='group', right_on='group')
    predicted_df.insert(0, 'state', 'CT')
    predicted_df.to_csv(predictions_data_path_cumulative, index=None)

    # print("before scatter")
    # print(X_data.head())
    scatter_plot_cluster(X_data, Y_data)


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




if __name__ == '__main__':
    # ct_cumulative_counts()
    df = pd.read_csv(raw_data_path)

    X_data = df.loc[0:, ['case_counts']]


    # scaler = StandardScaler()
    # X_data = pd.DataFrame(column = {'case_counts'scaler.fit_transform(X_data))
    # print("after scaling \n", X_data.head())
    elbow_plt(X_data)

    k = 6
    kmeans_model_predict_cumulative(X_data, k)
    # as per above fig it should be k=3






