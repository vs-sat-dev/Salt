# Salt

I running a custom Unet neural network to make image segmentation on TGS Salt dataset.

I wrote the code and made only one change - I changed the learning rate from 0.001 to 0.0001 to get the following results from the validation dataset.

image|mask|prediction
:----|:--:|:---------
![](readme_images/image1.png)|![](readme_images/mask1.png)|![](readme_images/pred1.png)
![](readme_images/image2.png)|![](readme_images/mask2.png)|![](readme_images/pred2.png)
![](readme_images/image3.png)|![](readme_images/mask3.png)|![](readme_images/pred3.png)
![](readme_images/image4.png)|![](readme_images/mask4.png)|![](readme_images/pred4.png)
![](readme_images/image5.png)|![](readme_images/mask5.png)|![](readme_images/pred5.png)
![](readme_images/image6.png)|![](readme_images/mask6.png)|![](readme_images/pred6.png)
![](readme_images/image7.png)|![](readme_images/mask7.png)|![](readme_images/pred7.png)
![](readme_images/image8.png)|![](readme_images/mask8.png)|![](readme_images/pred8.png)
![](readme_images/image9.png)|![](readme_images/mask9.png)|![](readme_images/pred9.png)
![](readme_images/image10.png)|![](readme_images/mask10.png)|![](readme_images/pred10.png)
![](readme_images/image11.png)|![](readme_images/mask11.png)|![](readme_images/pred11.png)
![](readme_images/image12.png)|![](readme_images/mask12.png)|![](readme_images/pred12.png)
![](readme_images/image13.png)|![](readme_images/mask13.png)|![](readme_images/pred13.png)
![](readme_images/image14.png)|![](readme_images/mask14.png)|![](readme_images/pred14.png)
![](readme_images/image15.png)|![](readme_images/mask15.png)|![](readme_images/pred15.png)
![](readme_images/image16.png)|![](readme_images/mask16.png)|![](readme_images/pred16.png)
![](readme_images/image17.png)|![](readme_images/mask17.png)|![](readme_images/pred17.png)
![](readme_images/image18.png)|![](readme_images/mask18.png)|![](readme_images/pred18.png)
![](readme_images/image19.png)|![](readme_images/mask19.png)|![](readme_images/pred19.png)
![](readme_images/image20.png)|![](readme_images/mask20.png)|![](readme_images/pred20.png)
![](readme_images/image21.png)|![](readme_images/mask21.png)|![](readme_images/pred21.png)
![](readme_images/image22.png)|![](readme_images/mask22.png)|![](readme_images/pred22.png)
![](readme_images/image23.png)|![](readme_images/mask23.png)|![](readme_images/pred23.png)
![](readme_images/image24.png)|![](readme_images/mask24.png)|![](readme_images/pred24.png)
![](readme_images/image25.png)|![](readme_images/mask25.png)|![](readme_images/pred25.png)
![](readme_images/image26.png)|![](readme_images/mask26.png)|![](readme_images/pred26.png)
![](readme_images/image27.png)|![](readme_images/mask27.png)|![](readme_images/pred27.png)
![](readme_images/image28.png)|![](readme_images/mask28.png)|![](readme_images/pred28.png)
![](readme_images/image29.png)|![](readme_images/mask29.png)|![](readme_images/pred29.png)
![](readme_images/image30.png)|![](readme_images/mask30.png)|![](readme_images/pred30.png)
![](readme_images/image31.png)|![](readme_images/mask31.png)|![](readme_images/pred31.png)
![](readme_images/image32.png)|![](readme_images/mask32.png)|![](readme_images/pred32.png)

I didn't try to change neural network architecture and hyperparameter tuning and because of that, the results could be better.
