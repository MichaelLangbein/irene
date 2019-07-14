try with a pretrained network
try without pooling after every conv
try some more data augmentation





try to predict with simple two layer fully connected. 
try with a lot more imput data


storms labeled by globally worst
        small model 
                overfits ... 1560834294
        reasonable model 
                overfits ... 1560804002 
        large model
                converges to mean ... 1560834525
        pretrained net
        twobranch-net: one conv-branch and one label-GRU-branch
storms split by sliding window, labeled by last




peak of a storm might be somewhere in middle. 
Should we split storms with a sliding window?


My model is just learning the mean occurrence prob of each class!
This is because it cannot extract useful information from the input data.
And that is because I dont separate the storms good enough.

    Changed by using SGD. Now finding a good minimum. 
    But: generalizes very badly. 

        Try with dropout
