from sklearn.model_selection import train_test_split
from run import run
import os


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    images, masks = [], []
    for image in os.listdir('images'):
        images.append(f'images/{image}')
    for mask in os.listdir('masks'):
        masks.append(f'masks/{mask}')

    images_train, images_test_val, masks_train, masks_test_val = train_test_split(images, masks,
                                                                                  test_size=0.25, random_state=0)
    images_test, images_val, masks_test, masks_val = train_test_split(images_test_val, masks_test_val,
                                                                      test_size=0.2, random_state=0)

    run(images_train, masks_train, images_test, masks_test, images_val, masks_val)

