def cumulative_ct_covid_data():
    data_list=[]
    today = datetime.strftime(datetime.today(),"%m%d%Y")[1:]
    #url="https://portal.ct.gov/-/media/Coronavirus/CTDPHCOVID19summary9112020.pdf"
    url="https://portal.ct.gov/-/media/Coronavirus/CTDPHCOVID19summary{}.pdf".format(today)
    print(url)
    ###########################################
    #uncomment  below lines for today's gove data
    pdf_data = requests.get(url)
    pdf_content = BytesIO(pdf_data.content)
    reader = PdfFileReader(pdf_content)
    ###################################
    #REmove below 2 lines for actual today's gov data"
    #pdffile=open(r"c:\users\babub\downloads\CTDPHCOVID19summary9112020.pdf","rb")
    #reader = PdfFileReader(pdffile)
    ###############################################
    for eachpage in range (4,5):
        try:
            case_count_page = reader.getPage(eachpage).extractText()
            if len(re.findall("Cumulative Number of COVID",case_count_page)) > 0:
                case_counts = re.findall("Andover[\s\S]+Preston",case_count_page)[0]
                #city_counts = re.findall("\S+[\n ]+[\n]+\S+[\n ]+[\n]+\S+[\n ]+[\n]+",case_counts)
                city_counts = re.findall("\S+?\s*\S*[\n ]+[\n]+\S+[\n ]+[\n]+\S+[\n ]+[\n]+",case_counts)
                for each_city in city_counts:
                    try:
                        int(re.split("[\n ]+[\n]+",each_city)[1])
                        data_list.append(re.split("[\n ]+[\n]+",each_city)[0:3])
                    except:
                        pass
                
                df =pd.DataFrame(data_list,columns=["city","case_counts","probable_counts"])
                df.to_csv(raw_data_path,header=True,index=None)
        except Exception as e:
            print("Exception occurred",e)