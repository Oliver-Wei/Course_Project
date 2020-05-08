# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example():
	# load the image
    model = load_model('final_model.h5')
    f = open("../data/form/labels.txt","w+")
    y = [0,2,1,3,4,6,5,5,7,8,9,9,8,9,2,5,7,6,0,1,4,3,6,8,0,2,1,3,4,4,5,7,6,8,5,9,2,1,2,4,1,7,7,5,3,6,6,7,1,1,2,2,2,1,6,8,5,3,4,7,2,2,1,6,4,5,5,3,8,1,5,3]
    acc = 0
    predicted_digits = []
    for i in range(72):
        img = load_image('../data/form/output%d.jpg' % i)
        # load model
        # predict the class
        digit = model.predict_classes(img)
        predicted_digits.append(digit[0])
        if (digit[0] == y[i]):
            acc += 1
        else:
            print("The %dth predicted number is " %i, digit[0]," and the correct number is ",y[i])
        # print(digit[0],end = " ")
    f.write(str(predicted_digits))
    f.close()
    print("\n",acc/72.0)
# entry point, run the example
run_example()