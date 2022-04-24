
# =============================================================================
# Prediction
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.models import  load_model
from gtts import gTTS
from playsound import playsound
Classes = ["Ten Pounds","100 Pound","Twenty Pounds","200 Pound","Five Pounds","50 Pound"]
# Pre-Processing test data same as train data.
size  =124

def prepare(img_path):
    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)

model = load_model('VGGXRAY.h5')

result = np.argmax(model.predict([prepare('data/test/200/200.0.jpg')]),axis=-1)
coin=image.load_img('data/test/200/200.0.jpg')
myobj = gTTS(text=Classes[int(result)], lang='en', slow=False)
myobj.save("currancy.mp3")

plt.imshow(coin)
print (Classes[int(result)])
playsound("currancy.mp3")
