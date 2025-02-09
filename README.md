# CP217-PROJECT-1-ARE-YOU-IN-A-SAFE-BUILDING?
The goal of this project is to develop machine learning models to classify Google Street View images of buildings into five classes: Steel, Concrete, Masonry, Wooden, and Steel with panel buildings.  
![image](https://github.com/user-attachments/assets/5d0f0360-4480-4381-b68a-5de5b8df7e8d)
![image](https://github.com/user-attachments/assets/02552801-2c41-445f-979e-7597206ffb5d)
![image](https://github.com/user-attachments/assets/57ead882-5d39-4e66-b565-bb862e29b741)

From this project/assignment we were able to understand and demonstrate that the effectiveness of
classical Models like SVM increases when they are able to train on extracted/relevant features. This
needs further exploration from our side. Instead of just using the classic max voting ensemble we also
generated ensembles by averaging softmax-probability outputs of multiple models. Because of the
data being low-quantity and noisy and models showing high variance a stacking classifier that takes
output of multiple models and makes a more sophisticated ensemble is also something we thought
of implementing.
