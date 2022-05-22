# SignLanguageRecognition

## 1. Datensatz bekommen
- Minimalanforderung: 2 - 3 verschiedene Gesten
- convinient: gelabelte Daten über API
- Bsp: Chalern

--> https://chalearnlap.cvc.uab.cat/dataset/40/description/ NAME: musterkek PW:3-_QfQXGTTJt8a#W


oder--> https://github.com/dxli94/WLASL American(scrapes webpages)
## 2. Datensatz verfremden/transformieren - "Aufblähen"
- Stichwort "augmented data"
- verändern von: Helligkeit, Farben, Rotation, Translation
- Kombinationen möglich

### 2.1 Skelletierung auf dem Datensatz anwenden (Optional)
- Eingabe: Video, Ausgabe: Video
- Minimalanforderung: Hand skelletiert
- Idealvorstellung: Skelletierung des gesamten Körpers
- Datensatz aus Punkt 2 benutzen
- [Mediapipe holistic](https://google.github.io/mediapipe/solutions/holistic.html)

## 3. Abstrahierung der Videodaten auf ein Bild
### Strategie 1: OPTiCAL flow
- durch arithmetisches Mittel und Varianz
### Strategie 2: Skelletpunkte einfärben
- vordef. Farbe für jeden Skelletpunkt.
- Normierung der Skellette (Körpergröße/Handgröße/Entfernung zur Kamera - irrelevant)
- Farbhelligkeit korrelliert mit Zeit (umso später, umso dunkler)
- arithmetisches Mittel aus den "bunten Skelletten"

#### *Teilergebnis: Gelabelter Datensatz aus abstakten Bildern mit Gestenwort.*

## 4. KNN zur Klassifikation erstellen

### 4.1. zusätzliche Merkmale als Eingansgrößen definieren (Optional)
- zusätzlich zu den OPTiCAL flow Bildern
- Varianz, Standardabweichung, ...

### 4.2. yolo4 CNN anlernen
- evtl. Mehrschichtig mit Merkmale aus 4.1
- yolo4 ist bereits angelernt! Anpassung der letzten Schichten (Gewichte entfernen)
