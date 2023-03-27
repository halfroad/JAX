import jax
from PIL import Image

def pixel_array(imageName) -> jax.numpy.array:

    image = Image.open(imageName)
    
    print("image.format = ", image.format)
    print("image.size = ", image.size)
    print("image.mode = ", image.mode)
    
    array = jax.numpy.asarray(image)
    
    return array
    
def test():

    imageName = "../../Shares/Images/Colorful.jpeg"
    array = pixel_array(imageName)
    
    print("array.shape = ", array.shape)
    print("array[: 5] = ", array[: 5, : 5])
    print("array = ", array)

if __name__ == "__main__":
    
    test()
             
             
