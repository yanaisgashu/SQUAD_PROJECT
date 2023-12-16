# SQuAD-Project
ML Analysis of the Stanford Question and Answer Dataset



## Introduction
With recent innovations circling around the new transformer model, Question-Answer based systems have been an integral part of our everyday lives, especially in this time and age that we live in where these  systems have almost acquired human level comprehensive skills. With the SQuAD 1.0 dataset there were contexts, questions and answers to these questions, but with the recent SQuAD 2.0 dataset we have a plethora of questions with answers alongside unanswerable questions. A lot of times, the usage of chatbots has come with a certain drawback, which is if it does not know the answer it will still provide you an answer, THE WRONG ANSWER. So telling whether a question is unanswerable is something that has proved invaluable. We plan on fine-tuning the BERT model on the SQuAD 2.0 dataset, and using it to test on how it does on determining whether an adversarial question, given a context, is answerable or unanswerable. 

## Problem Definition
The problem here is that, whenever a user uses any AI chat bot like Chat GPT 3.5 or Jasper, whenever a question is unanswerable, an answer is still given even though it is completely wrong. This will lead to misinformation being spread and problems arise from there. Simply, our project would just eliminate that problem by explaining to the user that an answer could not be provided.

## Methods
In order to preprocess the data we used the tokenizer from the Bert model in order to transform the sentences into data points. We then used the PCA algorithm to reduce the data down to two and three dimensional spaces so we could run kmeans on it and display the results on it in visual form. Following this we ran PCA with components ranging from 1 to 50 to determine how much of the variance was captured across the range. Lastly we split our data to  train and test a logistic regression model and it performed with 65% accuracy rate.

## Results and Discussion
### EDA
<img width="739" alt="Results11" src="https://github.gatech.edu/storage/user/55316/files/578bd817-d9e3-4d0d-8fac-a5444125b00e">

As seen in the code we have in our repo, this is the first graph that we have made. We thought that it would be a great idea to fully understand what our data looks like and how it is distributed in order to understand the results that we may get. The above bar graph titled ‘Distribution of Question Types” shows the frequencies of the most common types of questions that prevailed in the dataset we had, as seen there is a disproportionately large amount of what questions, this could just be that “what” questions are easily verifiable when it comes to whether or not a question is answerable vs unanswerable. But we are a bit afraid that the over representation of a single question type could create implicit bias in our algorithm causing overfitting on the questions which might lead to inaccurate results.

<img width="1010" alt="Results13" src="https://github.gatech.edu/storage/user/55316/files/1f31e1fa-48e0-4f88-b47f-051dc08695e1">

The graph above shows the distributions of the lengths of questions and the lengths of contexts. We have a good almost bell curve pattern shown here as the frequency peaks around the middle and tapers off to the right and left. It is good that there is a bit of variance in the distribution of the lengths for both the context and the questions.

<img width="545" alt="Results14" src="https://github.gatech.edu/storage/user/55316/files/0b655f63-609e-4f08-98d7-a0420919a58a">

The above graph is a pie chart of the amount of Answerable questions present in the dataset as opposed to the unanswerable questions. Wait? 66.6 percent answerable questions and 33.4 percent unanswerable questions? That does not seem right? I did learn about this in my machine learning class, what did they call it, overfitting? That was the first thought that struck us looking at this atrocious 2:1 distribution. And it does make sense If we have more answerable examples in our dataset as opposed to the unanswerable ones, the model will do better in the most prevalent class and perform suboptimally in the less prevalent class. A phenomenon coined by the term “class imbalance”. Now with further investigation, although this will make it hard for the model to have a high accuracy measure, that appears to be precisely the case. The dataset was designed to make it harder to distinguish between these two classes. If a high accuracy can be attained in this dataset that suffers from some class imbalance, that might as well be a good leap for in the step of creating reliable AI systems. From an intuitive perspective, doing good in harsh conditions means doing even better when conditions are not as harsh.

#### PCA (Does it really help here)
If noted from our code, we have used the BERT model tokenizer to then convert our questions into vectors in the embedded space. These vectors in this embedded space have hundreds of dimensions. This sounds intimidating, and at first we decided to jump the gun and reduce the number of dimensions to two principal components, and then we decided to do some k-means and DBSCAN on the data after we have applied PCA to it, to see any naturally forming clusters. And the graph below was what we got.

<img width="784" alt="Results16" src="https://github.gatech.edu/storage/user/55316/files/bbb9e72e-f96f-452e-be3b-ad729c1912fe">
<img width="544" alt="Results9" src="https://github.gatech.edu/storage/user/55316/files/907dea7f-b172-47a1-9975-cbfb00d37cdb">

DBSCAN:
For DBSCAN, we used the elbow method to determine the best epsilon to use, with minimum points equal to twice the number of features. DBSCAN provides us a clustering of similar sentences or words, thus giving us an idea of how a question with an answer might differ from a question without an answer. Given the sample dataset, DBSCAN didn’t turn out as useful as hoped. The sample points weren’t diverse enough to different density areas. Perhaps with more sample points the clustering will provide a better result.

<img width="467" alt="Results8" src="https://github.gatech.edu/storage/user/55316/files/74ae3fa1-097e-4ff5-84aa-0a792e92d729">
<img width="682" alt="Results17" src="https://github.gatech.edu/storage/user/55316/files/38eaf64a-5e42-407c-9ad8-d68433ca6892">
<img width="542" alt="Result18" src="https://github.gatech.edu/storage/user/55316/files/4ce78db7-cc54-43fa-bfbe-5339c28bda37">

Note that as seen in our code, these are just some random samplings of the data that we have plotted as the entire data would by too much to be plotted and understood. The amount of variance that was kept from these clusters was not good at all, and it had us considering why for a little while. This is a table of the results we got.

<img width="856" alt="Results5" src="https://github.gatech.edu/storage/user/55316/files/8d2d19b0-1959-4f18-bf6f-9ca3762ef8c6">

We can see that With two components 22.65 percent of the variance was explained and with 3 components 28.11 percent was explained. This does not look promising, and training our classification model on this can not give us a good reason. We were astounded as to why this was happening, but even more so curious as to how many components it would take to have a good amount of explained variance, and so we decided to see what happens as we increase the principal components, and the table and graph below does a great job of showing what happens.

<img width="719" alt="Results6" src="https://github.gatech.edu/storage/user/55316/files/93712c1e-185e-477b-bbd8-86a78462a796">
<img width="674" alt="Results7" src="https://github.gatech.edu/storage/user/55316/files/a1d3c0a6-151d-46f1-bdb1-62985e099f6d">

As seen above if we had a 90 percent cut off threshold the number of components we need is a little over 30. That's a lot to visualize as we unfortunately live in a 3D world. With further exploration we were able to see that the reason why 2 and 3 components did not keep as much explained variance and we needed way more principal components to have better representation of our data has to do with how PCA works. PCA is very linear. But after we used the BERT tokenizer and then convert our questions into the embedded space, the BERT model does a forward pass through the trained neural network, so although this vector has a really high dimension, every dimension captures an important relationship between words, that makes it very important, and this process is very non linear, so that explains why we needed to have a lot of principal components before we good a good amount of explained variance. This means that training a model on the 2 or 3 principal components is not a good idea. But the PCA can still be paired with a clustering algorithm like k means or DBSCAN to capture any naturally forming clusters. 

### Logistic Regression
The table shows that currently we have an accuracy of 65.5 percent when we ran a logistic regression algorithm. This is a moderate level of accuracy but there is room for improvement. This is a very simple model, and it is linear. We are thinking that working with a model that is more sophisticated that can capture the nonlinear relationship in the embedded space is a good start. We have a lot of ideas for improvement Such as using models like support vector machines with non-linear kernels, random forests, and finally using neural networks. It will be very interesting to see what these models show us. 

<img width="422" alt="Results2" src="https://github.gatech.edu/storage/user/55316/files/84703ed9-eaf7-43ca-b99d-fe9db7d3735e">

The kmeans illustration showed some possible trends and connections in the data possibly the division into certain types of questions. Using the graph of PCA explained variance compared to number of components we determined that a PCA reduction to 33 components captured 90% of the variance of the data. Our trained logistic regression model performed with 65% accuracy which is significantly less than optimal but a good start. Proceeding into the future we will try different methods that incorporate regularization to control for possible noise in the data that is stopping linear regression from performing optimally. We may also go back and implement more data preprocessing techniques to see if we can improve accuracy this way as well.

### Next Steps
Proceeding into the future we will try different methods that incorporate regularization to control for possible noise in the data that is stopping linear regression from performing optimally. We may also go back and implement more data preprocessing techniques to see if we can improve accuracy this way as well.

## References
- [Adversarial Examples for Evaluating Reading Comprehension Systems; Robin Jia, Percy Liang](https://arxiv.org/abs/1707.07328)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding; Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova](https://arxiv.org/abs/1810.04805)
- [Know What You Don't Know: Unanswerable Questions for SQuAD; Pranav Rajpurkar, Robin Jia, Percy Liang](https://arxiv.org/abs/1806.03822 )


### End of Midterm report (for regrade purposes)

### Transfer Learning Using BERT MODEL
It was apparent that a simple logistic regression could by no means capture the nuances and complexity of our model. The embedding space we had had hundreds of dimensions, and so we knew that it was a good idea to implement a neural network. We decided to use a pretrained BERT model for our architecture, and we finetuned the BERT model to have an additional layer coupled with an activation function(sigmoid), and in the last hidden layer of the architecture we introduced a dropout regularization techniquie with probability for dropout of 0.3, which we found was suitable hyperparameter. This allowed us to make sure that there would be no overfitting. This was a challenging task as the SQUAD 2.0 data had an increasingly difficult amount of adversarial contexts making it imposible to implement a heuristic that could give a high accuracy. But our finetuned model was able to achieve a 72 percent ROC under the curve as shown below. And we were able to have an accuracy of about 67 percent. Which given the nature and level of the intensity in the adversarial contexts that were unanswerable was a good result, the 72 percent ROC indicating that given a positive and negative example the model will be able to distinguish between them 72 percent of the time. Below is the plot ofthe accuracy we had 


### Random Forest
In order to determine the most effective implementation we tested different values for different parameters and graphed the results of these different values to look for correlation as well as the optimal value. Our data contains hundreds of dimensions but the ensemble nature of trees didn't neccesarily mean that we needed to use a high number features to get optimal results so we tested a wide range of values varying the number of trees from 25 - 1000 and the max number of features from 10 - 100. The results were all over the place. There didn't appear to be a strong amount of correlation but certain values were clearly better than the rest for number of trees 200 produced the best results and for max features 65 was optimal.

<img width="735" alt="Screenshot 2023-12-04 at 12 58 20 AM" src="https://github.gatech.edu/storage/user/55316/files/e31a6650-30b9-4e01-9837-38f27816cbc2">
<img width="735" alt="Screenshot 2023-12-04 at 12 58 40 AM" src="https://github.gatech.edu/storage/user/55316/files/fd3da18d-6841-4025-b602-21c9d79ed5d2">

Given these results we determined that the ideal number of trees was 200 and the ideal number of features and max depth was 65. This didn't neccesarily mean that together would produce the best random forests but after testing several possibble combinations of these parameters it turned out that 200 trees and 65 max features was the best. We used these numbers to fit the final parameter optimized random forest to our data and got the following results.

<img width="388" alt="Screenshot 2023-12-04 at 1 06 21 AM" src="https://github.gatech.edu/storage/user/55316/files/c3ffa146-21e0-402b-aa52-d8d8f2826fe0">

Our parameter optimized random forest yielded an accuracy of .6925 which was the best result out of all the methods of classification we had tried up into this point. We still had the neural net and SVM to implement and were hoping these models would produce an even better accuracy score.

### SVM
SVM was the most hopeful supervised algorithm. Due to the high dimensionality of our data and its classification into two categories, SVM seemed to be the best choice algorithmically. We reduced the dimensionality of our data through PCA and then ran SVM on the reduced data sets. We chose to first try a smaller sample size of 500 data points, to both debug and analyze the possible results of SVM. This resulted in an accuracy of 0.74, the highest accuracy of the project. However, we wanted to make sure that the accuracy would hold with a larger sample size since there was an imbalance in the categorization of data between answerable and unanswerable. For a second run, the sample size was 20% of the total data, resulting in an accuracy of 0.676.

<img width="439" alt="Screenshot 2023-12-05 at 1 32 40 PM" src="https://github.gatech.edu/storage/user/55316/files/20c77567-ae9f-4623-9756-23c137687b81">
<img width="441" alt="Screenshot 2023-12-05 at 1 33 11 PM" src="https://github.gatech.edu/storage/user/55316/files/17676eb0-242d-4bf4-af07-797538b54be5">
<img width="439" alt="Screenshot 2023-12-05 at 1 32 40 PM" src="https://github.gatech.edu/storage/user/70667/files/1b4c2cf0-6e56-4552-b1cb-aee1d6bdd3a8">
<img width="687" alt="Screenshot 2023-12-05 at 2 27 07 PM" src="https://github.gatech.edu/storage/user/70667/files/38c54be4-2e57-4555-9958-adcbd9fe2a28">


### Ensemble Learning (Stacking, THE FINAL META LOGISTIC )
So far we have three algorithms, SVM, Random Forest, and a Neural Network architecture. Finally we wanted to combine these models into one model which had the potential of leveraging the positive aspects of these models, and so we decided to try out an Ensemble Learning technique. We used the Stacking technique to be more specific. We combined our models into a bigger meta model, and this meta model was a logistic Regression Algorithm. With the Ensemble learning technique we were able to aquire an accuracy of 66 percent. Here is a plot of the Ensemble Learning accuracy we were able to acquire. 


### Final Note
No doubt, we were able to see the difficulty in the problem of being able to distinguish between answerable and unanswerable questions, given some extremely adversarial unanswerable questions. We embarked on this task due to its real life use cases as discussed in the introduction. Although we were able to get only close and around 70 percent accuracies even while using the finetuned BERT architecture, we are very pleased with the results we have recieved due to the difficulty of the problem. State of the art models were only able to receive accuracies in the 60 percents in the SQUAD 2.0 data. All in all, with the optimistic results we were able to receive from our 4 supervised and deep learning algorithms we were able to implement (SVM, Random Forest, Neural Network (Transfer Learning on BERT model), and Ensemble Learning) alongside the numerous unsupervised techniques we implemented for EDA such as GMM, Kmeans and PCA, we are happy to have been able to have trained our models successfully.
