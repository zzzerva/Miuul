#sanal ortamların listelenmesi
#conda env list

#sanal ortam oluşturma
#conda create -n myenv(isim)

#saanl ortamı aktifleştirme
#conda activate myenv

#deactive
#To deactivate an active environment

#yüklü paketlerin listelenmesi
#conda list

#paket yükleme
#conda install numpy(paket-adı)

#aynı anda birden fazla paket yüklenmesi
#conda install numpy scipy pandas

#paket silme
#conda remove package-name

#versiyen seçme
#conda install numpy=1.20.1

#güncelleme
#conda upgrade numpy
#conda upgrade -all

#pip: pypi (python package index) paket yöneim aracı

#pip install paket-adi
#pip install paket-adi==1.2.1

#conda env export > environment.yaml (paketler dışarı aktarmak için)

#sanal ortam silme
#conda env remove -n myenv

#conda env create -f enviroment.yaml (sanal ortamı canlandırma)