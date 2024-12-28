from numpy.core.defchararray import upper

students = ["erva", "seymq", "mihri", "göko"]

students[0]

for i in students:
    print(i)


for i in students:
    print(i.upper())

maas = [1000, 2000, 3000, 4000, 5000]
for i in maas:
    print(int(i*20/100 + i))

def new_salary(maas, zam):
    return int(maas * zam / 100 + maas)

new_salary(2000, 10)

for i in maas:
    if i >= 3000:
        print(new_salary(i, 10))
    else:
        print(new_salary(i, 20))

len("erva")
range(len("erva"))
help(range)


def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("merhaba ben erva nasılsın")

liste = ["erva", "seyma", "mihr", "göko"]

for student in liste:
    print(student)

for index, student in enumerate(liste):
    print(index, student)

liste = ["erva", "seyma", "mihr", "göko"]
A = []
B = []

for index, student in enumerate(liste):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

liste = ["erva", "seyma", "mihr", "göko"]
def divide_student(liste):
    groups =[[], []]
    for index, ogrenci in enumerate(liste):
        if index % 2 == 0:
            groups[0].append(ogrenci)
        else:
            groups[1].append(ogrenci)
    print(groups)
    return groups

a = divide_student(liste)
a[0]
a[1]

