import csv

with open('SQuAD_csv_shuf.csv', 'r', errors='ignore') as csv_file: 
    csv_reader = csv.reader(csv_file, delimiter=',')
    count = 0
    with open('squad_eng.csv', 'w', newline='') as write_file:
        squad_writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            try:
                question = row[2]
                text = "<answer> " + row[5] + " <context> " + row[1]
                print(question + " : " + text)
                squad_writer.writerow([question, text])
                count += 1
            except Exception as e:
                print("Warning: Got some error, skipping line")
                print(e)
            if count > 27000:
                break