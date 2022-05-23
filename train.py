from preprocessing.dataretrieval import *
from preprocessing.featureextraction import *

#training main --> MODEL &&& testing vom CNN -- exportiere Model
#irgendwoanders visualize main (Model,webcam)--> lädt model und wertet aus und visualisiert
if __name__ == "__main__":
    #clean data and put it into /data/train/
    dr = Dataretrieval(CleanCHALearn())
    dr.retrieve(["breakfast", "hungry"])
    fe = FeatureExtraction(Skelleting_as_image())
    fe.extractFeature("data/raw")



