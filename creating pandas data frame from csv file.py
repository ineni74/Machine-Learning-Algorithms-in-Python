import csv
csv_data = []
with open("Ecommerce Customers") as csvfile:
    csvReader = csv.reader(csvfile, delimiter=',')
    for row in csvReader:
        csv_data.append(row)
        print(row)
data_df = pd.DataFrame(csv_data[1:], columns=csv_data[0])