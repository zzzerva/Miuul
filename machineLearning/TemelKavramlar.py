#Bilgiasayrların insanlara benzer şekilde öğrenmesini sağlamak için çeşitli algoritmaların
#geliştirilmesi için çalışan bilimsel çalışma alanıdır


###Öğrenme türleri 3e ayrılır.
#Pekiştirmeli Öğrenme: Deneme yanılma yoluyla olan öğrenmedir.
#Denetimli Öğrenme: Üzerinde çalışan verilerde bağımlı değişken(Target) varsa bu denetimil öğrenme olduğu anlamına gelir
#Denetimsiz Öğrenme: Bağımlı değişken yoksa denetimsi öğrenmedir


###Problem Türleri
#Regresyon problemiyse bağımlı değişken sayısaldır.
#Sınıflandırma Problemi ise bağımlı değişken kategoriktir

#MSE = 1/n ∑ (yi - Yi)^2 (tavan n taban i = 1) (Ne kadar küçük o kadar iyi)
#Gerçek değerlerle tahmin edilen değerlerin farkının karesinin toplamı

#RMS = √ 1/n ∑ (yi - Yi)^2 (kare alma işleminin geri dönüşü)

#MAE = 1/n ∑ |yi - Yi| (farkların mutlakları alınır)

#ACCURACY = (Doğru Sınıflandırma Sayısı) / (Toplam Sınıflandırılan Gözlem Sayısı) (ne kadar yükeke o kdar iyidir)

####Model Doğrulama Yöntemleri

#Holdout(Sınama seti yaklaşımı): Orjinal veri setini eğitim seti ve test seti olmak üere ikiye ayırırız

#K Fold Cross Validation(K-Katlı Çapra Doğrulama): Eğitim setini 5e böl her işlemde 4üyle model oluştur 1iyle test et

###Yanlılık-Varyans Değiş Tokuşu(Bias-Variance Tradeoff)

#Overfitting: Aşırı öğrenme(Yüksek Varyans), Underfitting: Az öğrenme(Yükek Yanlılık)

#Eğitim seti ve test seti hata değişimleri incelenir. Bu iki hatanın birbirinden ayrılmaya başladığı nokta aşırı öğrenmenin başladığı nokta olarak kabul edilir.

#Model Karmaşıklığı: Modelin hassaslaştırılması

#Doğrual Regreyon: Grafikte çizilecek olan doğrunun formülü [Cost(b, w) = 1/2 ∑ ((b + wxi) - yi)^2 dir. ((b + wxi) = Yi)

#####
##LOJİSTİK REGRESYON
#####

#Accuracy: (TP+TN) / (TP+TN+FP+FN)
#Precision: Pozitif sınıf tahmininlerinin başarı oranı TP / (TP+FP)
#Recall: Pozitif sının doğru tahmin edilme oranıdır TP / (TP+FN)
#F1 SCORE: 2*(Precision*Recall)/(Precision+Recall)
#Threshold artarsa accuracy düşer, threshold azalırsa acccuracy artar
#ROC Curve(ROC eğrisi)
#ROC eğrisinin altında kalan alanın integrali alınırsa AUC(area under curve) elde edilir

#LOG Loss
#Entropi yüksekse çeşitlilik azdır(daha iyi)
#Probability 0,80 ve 0,48 olsun Log loss hesabında -1*log(0,80) < -1*log(0,48) olur

