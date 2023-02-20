import csv


age_intercept = 23.2188604687

def load_age_lexica(file_name="emnlp14age.csv"):
    age_lexica={}
    with open(file_name) as infile:
        reader=csv.DictReader(infile)
        for data in reader:
            weight = float(data['weight'])
            term= data['term']
            age_lexica[term]=weight

        del age_lexica['_intercept']
        return age_lexica
    
age_lexica = load_age_lexica()

def get_age(text):
	words = text.split()

	text_scores = {}
	for word in words:
		text_scores[word] = text_scores.get(word, 0) + 1

	age = 0
	words_count = 0
	for word, count in text_scores.items():
		if word in age_lexica:
			words_count +=count
			age += count * age_lexica[word]

	age = age / words_count + age_intercept
	
	return age