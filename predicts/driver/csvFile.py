import csv
import pandas
import os
def generateCsv(fileName):
    source = "input files/test/{}"
    files    = [os.path.join(source,f) for f in os.listdir(source)]

    # out = csv.writer(open(fileName, "w"), delimiter=',', quoting=csv.QUOTE_ALL)
    # out.writerow("{},{},{}".format("fname", "gender", "region"))
    # for f in files
    #     out.writerow("{},{},{}".format(f,"",""))

def testCsv(fileName):
    print(fileName)
    print(os.getcwd())
    df = pandas.read_csv(fileName)
    # data = df.iloc[10, "test"]
    # data.loc[data['id'] > 2000, "first_name"] = "John"
    # df.loc[0,"test"] = "female_north"
    sourcepath = "test/"
    files = [os.path.join(sourcepath, f) for f in df["fname"][:10]]
    print(files)

    # df.to_csv("output_filename.csv", index=False, encoding='utf8')


testCsv("../../sources/test_metadata.csv")
# generateCsv("input files/Ky/testData.csv")