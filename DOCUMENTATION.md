# Penjelasan Singkatan Atribut Dataset Obesitas

Dataset ini memiliki 17 atribut yang merepresentasikan data kebiasaan makan (*eating habits*), kondisi fisik (*physical condition*), dan latar belakang individu. Berikut adalah penjelasan lengkap dari masing-masing atribut beserta singkatannya:

## Atribut Profil & Fisik Dasar
1. **Gender**: Jenis kelamin (*Female* / *Male*)
2. **Age**: Usia dalam satuan tahun
3. **Height**: Tinggi badan dalam satuan meter
4. **Weight**: Berat badan dalam satuan kilogram
5. **family_history_with_overweight**: Apakah responden memiliki riwayat keluarga yang mengalami obesitas atau kelebihan berat badan (*yes* / *no*)

## Atribut Kebiasaan Makan (*Eating Habits*)
6. **FAVC** *(Frequent consumption of high caloric food)*: Kebiasaan sering mengonsumsi makanan berkalori tinggi (*yes* / *no*)
7. **FCVC** *(Frequency of consumption of vegetables)*: Frekuensi mengonsumsi sayuran dalam menu makan sehari-hari (Skala: 1 = Tidak pernah, 2 = Kadang-kadang, 3 = Selalu)
8. **NCP** *(Number of main meals)*: Jumlah waktu makan utama yang dilakukan dalam sehari (1 hingga 4)
9. **CAEC** *(Consumption of food between meals)*: Kebiasaan ngemil atau makan di antara waktu makan utama (*no*, *Sometimes*, *Frequently*, *Always*)
10. **CH2O** *(Consumption of water daily)*: Jumlah air mineral/air putih yang diminum dalam sehari (Skala: 1 = Kurang dari 1 liter, 2 = 1 hingga 2 liter, 3 = Lebih dari 2 liter)
11. **CALC** *(Consumption of alcohol)*: Frekuensi mengonsumsi minuman beralkohol (*no*, *Sometimes*, *Frequently*, *Always*)

## Atribut Gaya Hidup & Kondisi Fisik (*Physical Condition & Lifestyle*)
12. **SMOKE**: Apakah responden merupakan seorang perokok aktif (*yes* / *no*)
13. **SCC** *(Calories consumption monitoring)*: Apakah responden secara rutin memonitor/menghitung jumlah asupan kalori hariannya (*yes* / *no*)
14. **FAF** *(Physical activity frequency)*: Frekuensi melakukan aktivitas fisik atau berolahraga dalam seminggu (Skala: 0 = Tidak pernah, 1 = 1 hingga 2 hari, 2 = 3 hingga 4 hari, 3 = Lebih dari 4 hari)
15. **TUE** *(Time using technology devices)*: Waktu yang dihabiskan untuk menatap layar perangkat teknologi seperti *smartphone*, TV, atau komputer dalam sehari (Skala: 0 = 0-2 jam, 1 = 3-5 jam, 2 = Lebih dari 5 jam)
16. **MTRANS** *(Transportation used)*: Jenis moda transportasi utama yang sering digunakan sehari-hari (*Automobile*, *Motorbike*, *Bike*, *Public_Transportation*, *Walking*)

## Variabel Target / Kelas
17. **NObeyesdad**: Tingkat obesitas/level berat badan yang merupakan label klasifikasi utama dari dataset ini, dibagi menjadi 7 kategori:
    - *Insufficient_Weight* (Kekurangan berat badan)
    - *Normal_Weight* (Berat badan normal)
    - *Overweight_Level_I* (Kelebihan berat badan tingkat 1)
    - *Overweight_Level_II* (Kelebihan berat badan tingkat 2)
    - *Obesity_Type_I* (Obesitas tipe 1)
    - *Obesity_Type_II* (Obesitas tipe 2)
    - *Obesity_Type_III* (Obesitas tipe 3)
