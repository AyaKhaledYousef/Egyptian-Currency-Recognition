# =============================================================================
# 
# =============================================================================
import cv2
import os

class moneyPaper:
    def __init__(self,name,front,back):
        self.detector=cv2.xfeatures2d.SIFT_create()
        self.name = name
        self.front = front
        self.back = back

    def FrontH(self,trainKPFront,trainDescFront):
        self.trainKPFront,self.trainDescFront=self.detector.detectAndCompute(self.front,None)
        
    def BackH(self,trainKPBack,trainDescBack):
        self.trainKPBack,self.trainDescBack=self.detector.detectAndCompute(self.back,None)

    def matched_features(self,inDesc):
        matches=flann.knnMatch(inDesc,self.trainDescFront,k=2)
        goodMatch1=[]
        for m,n in matches:
            if(m.distance<0.6*n.distance):goodMatch1.append(m)
        matches=flann.knnMatch(inDesc,self.trainDescBack,k=2)
        goodMatch2=[]
        for m,n in matches:
            if(m.distance<0.6*n.distance):goodMatch2.append(m)
        return (max(len(goodMatch1),len(goodMatch2)),self)

    def mostlilkly(img_array,inDesc):
        possible=[(i.matched_features(inDesc)) for i in img_array]
        max=0
        result=""
        for j in possible:
             if(j[0]>max):
                 max=j[0]
                 result=j[1].name
        return result

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})
coins = [5,10,20,50,100,200]
img_array=[]
for i in coins:
    img1 = cv2.imread("data/train/money_"+str(i)+"_0.jpg",0)
    img2 = cv2.imread("data/train/money_"+str(i)+"_1.jpg",0)
    money = moneyPaper(str(i),img1,img2)
    img_array.append(money)
detector=cv2.SIFT_create()
trainKP = []
trainDesc= []
img_array2=[]
for i in img_array:
    trainKPFront,trainDescFront=detector.detectAndCompute(i.front,None)
    trainKPBack,trainDescBack=detector.detectAndCompute(i.back,None)
    i.FrontH(trainKPFront,trainDescFront)
    i.BackH(trainKPBack,trainDescBack)


for filename in os.listdir("data/test"):
    inp = cv2.imread("data/test/"+filename,0)
    inKP,inDesc=detector.detectAndCompute(inp,None)
    res = moneyPaper.mostlilkly(img_array,inDesc)
    print(filename,"\t\t is most likly \t\t\t"+(res)+" L.E")



# =============================================================================
# 
# =============================================================================
