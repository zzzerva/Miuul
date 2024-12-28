import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)
np.random.randint(0,10, size=10)
np.random.normal(10, 4, (4, 4))

a = np.random.randint(10, size=5)
a.ndim #boyut sayısı
a.shape #boyut bilgisi
a.size #toplam eleman sayısı
a.dtype #tip bilgisi


#reshape: boyut bilgisini yeniden şekillendirme
ar = np.random.randint(1,10, size=9)
ar.reshape(3,3)


#index seçimi (ındex section)

a = np.random.randint(10, size=10)
a[0]
a[0:5] #SOL DAHİL SAĞ  HARİÇ
a[0] = 999 #elemanı değiştirdik
#boyutlu arraylerde index saçimi nasıl yapılır

m = np.random.randint(10, size=(3,5))

m[0, 0] #satır, stün şeklinde sıralanır.
m[2, 3] = 999
m[2, 3] = 2.9 # float eklemesi yapsak da onu int şeklinde ekler
m[:, 0] #0. indisteki tüm satırlar
m[1, :] #1. indisteki tüm sütunlar
m[0:2, 0:3]


##
#Fancy index
##

v = np.arange(0, 30, 3) #0 dan 30 a kadar 30 hariç 3er 3er artan bir liste
v[1]
v[4]

catch = [1, 2, 3]
v[catch] #1. 2. ve 3. indisi getiriyor


##
#conditions on NumPy
##

v = np.array([1, 2, 3, 4, 5])

ab = []

for i in v:
    print(i)
for i in v:
    if i <3:
        ab.append(i)

v < 3 #listenin içindeki arrayleri sorgulaam yaparak boolean ifade döndürüyor

v[v < 3]

##
#mathematical operation
##

v / 5
v = np.array([1, 2, 3, 4, 5])
np.subtract(v, 1) #çıkarma
np.add(v, 1) #toplama
np.mean(v) #ortalama aldı
np.sum(v) #tüm elemanları topladı
np.min(v) #minim gösterdi
np.max(v) #maksimum gösterdi
np.var(v) #varyansını getirdi


#5*x0 + x1 = 12
#x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]]) #katsayılar
b = np.array([12, 10]) #sonuçlar

x = np.linalg.solve(a, b)
print(x)

a = 3
a**2