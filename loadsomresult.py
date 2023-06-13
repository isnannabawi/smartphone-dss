import csv
def loadsom(filename, remove_header):
	with open(filename, "r") as file:
		reader = csv.reader(file)
		data = list(reader)

	if remove_header:
		#menghapus header
		del data[0]

	# menulis nama kelas kedalam label
	label = [row[8] for row in data]
	label = [float(i) for i in label]

	# mengambil feature
	feature = [row[1:8] for row in data]
	feature = [[float(i) for i in row] for row in feature]

	length = len(label)

	return feature, label, length