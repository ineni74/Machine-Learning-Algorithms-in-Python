import csv
csv_data = []
with open("file_name") as csvfile:
    csvReader = csv.reader(csvfile, delimiter=',')
    for row in csvReader:
        csv_data.append(row)

data_df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
