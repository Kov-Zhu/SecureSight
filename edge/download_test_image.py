import urllib.request
from config import TEST_IMG_FLODER

url = "https://ultralytics.com/images/bus.jpg"
filename = TEST_IMG_FLODER + "test.jpg"
urllib.request.urlretrieve(url, filename)
print(f"The image has been downloaded as {TEST_IMG_FLODER}test.jpg")
