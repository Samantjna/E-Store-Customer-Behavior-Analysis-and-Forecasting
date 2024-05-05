# E-Parduotuvės Klientų Elgsenos Analizė ir Prognozavimas

### Projekto autorės:
- Samanta Čečkauskaitė
- Rasa Dzenkauskaitė
## Įvadas:
Analizuoti e-parduotuvės klientų pirkimo elgseną ir prognozuoti būsimus pirkimus, taip padedant verslui geriau suprasti klientų poreikius ir optimizuoti atsargų valdymą.


### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

### Technologijos:
Python ✦ TensorFlow/Keras ✦ Plotly/Dash ✦ scikit-learn ✦ Pandas

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

### Darbo etapai:
1.
2.
3.
4.

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
## 1.Duomenų valymas ir paruošimas naudojimui


## Duomenų analizė pagal klientus:

Pirmiausia analaziuojame kurios lyties klientai perka daugiausiai ir matome, kad abi lytis išleidžia panašią sumą pinigų.

| Nr |	Lytis |	Suma |
|-------|------|--------------|
|1  |Moteris	|342462421
|2	|Vyras	|338880262

Tada analizuojame klientų kiekį pagal amžiaus grupes ir matome, kad didžiausią mūsų klientų kiekį sudaro nuo 25 m. iki 50 metų grupės, o mažiausią kiekį sudaro klientai iki 25 metų.
Tai mums leidžią žinoti, kurio amžiaus žmonės apsiperka daugiausiai.

<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/047d2e07-10ef-4f3b-9cdb-0bb2fcf66d49" 
     width="480" 
     height="360" />

  |Klientų skirstymas|	Išleistos sumos vidurkis |
|-------|--------------|
|Klientai jaunesni nei 25 m.| 33671
|Klientai nuo 25 m. iki 50 m.|112988
|Klientai nuo 50 m.|94081

Analizuojame klientų pasirinkimą į prenumeratą/naujienlaiškį ir matome, kad tik 20% visų klientų renkasi gauti naujienlaiškį. Iš to galime spręsti, kad E parduotuvei nenaudinga yra šita paslauga.

<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/55907cdc-5cb3-43bf-81d5-2b47ce6172df"
     width="480" 
     height="360" />

| Naujienlaiškis |	Statistika |
|-------|------|
|Renkasi   |200126
|Nesirenka |49874

Klientų atsiskaitymo būdų analizė. Matome, kad klientai labiau linkė atsiskaityti Kreditine kortele 40% ir PayPal 30%. Siūlome reklamuoti, kad E parduotuvėje galima atsiskaityti šitais būdais.

<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/bb35c427-e60a-4e71-bc3b-cbd65127a23a"
     width="480" 
     height="360" />

| Mokėjimo metodas |	Statistika |
|-------|------|
|Credit Card   |100486
|PayPal          |74837
|Cash            |49894
|Crypto          |24783


## Duomenų analizė pagal pirkimus:

Labiausiai perkamos prekių kategorijos per visą laikotarpį. Matome, kad daugiausiai perka knygas ir rūbus, tad siūlytume juos reklamuoti daugiausiai ir turėti daugiau inventoriaus šiom kategorijom.

<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/b3d3172a-a558-4cfc-8d12-1e9aa4392d84"
     width="500" 
     height="360" />
     
| Nr. | Prekių kategorijos | Statistika 
|----|-------|------|
| 1. |Books          |223876
| 2. |Clothing       |225322
| 3. |Electronics    |150828
| 4. |Home           |149698

Apžiūrime pardavimus per mėnesį ir metus.


<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/8f327f1d-096d-4fb7-8ee9-f67ac3eb13d0"
     width="500" 
     height="360" />
<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/c3319852-2cc1-4584-8c2e-185d036cabaf"
     width="500" 
     height="360" />

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
## 2. Klientų segmentaciją su KMeans

Skirtingi klasteriai pagal spalvų juostą dešinėje atspindi kaip klientai gali būti grupuojami pagal išlaidų elgseną ir dažnumą, kas gali būti vertinga rinkos segmentavimo analizei. Iš to matome, kad yra trys klientų grupės atsižvelgiant į išleistą bendrą pinigų kiekį ir išleistą pinigų kiekį per pirkimą.

| Vertinimas. | Klasteriai | Gaunamas rezultatas 
|----|-------|------|
|K-Means silhouette score: |k= 3| 0.34
|K-Means silhouette score: |k= 6| 0.29
|K-Means silhouette score: |k= 9| 0.26
|Best silhouette score for k = 3

![image](https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/36a241dc-1434-4ff0-9f61-3e71e2eb0f85)

## 3. Modelio kūrimas






