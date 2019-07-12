try to predict with simple two layer fully connected. 
try with a lot more imput data


modelData is stored in h5 in order. 
Need to shuffle that through  - otherwise I only train model on early mornings storms, not on evenings storms


peak of a storm might be somewhere in middle. 
Should we split storms with a sliding window?


My model is just learning the mean occurrence prob of each class!
This is because it cannot extract useful information from the input data.
And that is because I dont separate the storms good enough.

    Changed by using SGD. Now finding a good minimum. 
    But: generalizes very badly. 

        Try with dropout