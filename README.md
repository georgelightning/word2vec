# Word2Vec from Scratch: The Brothers Karamazov

A pure NumPy implementation of the Word2Vec Skip-gram architecture with Negative Sampling (SGNS). This project maps semantic and philosophical relationships in Fyodor Dostoevsky’s *The Brothers Karamazov* without the use of high-level deep learning libraries.

## Technical Implementation
* **Architecture:** Skip-gram with Negative Sampling (SGNS).
* **Optimization:** Stochastic Gradient Descent (SGD) with a linear learning rate scheduler.
* **Initialization:** Xavier (Glorot) initialization for weight stability.
* **Negative Sampling:** Unigram table using the 3/4 power law for efficient selection (k=20).
* **Dimensions:** 30-dimensional embeddings.

## Training Parameters
* **Vocabulary Size:** ~4,350 unique tokens (min_count=5).
* **Window Size:** 4 (8 context words per center word).
* **Epochs:** 10.
* **Learning Rate:** Starts at 0.001, decaying linearly to 0.0001.

## Mathematical Objective
The model minimizes the Negative Log-Likelihood of the center-context word pairs:
$$L = -\log(\sigma(v_{pos} \cdot v_c)) - \sum_{i=1}^{k} \log(1 - \sigma(v_{neg,i} \cdot v_c))$$
### Gradients for Backpropagation

The derivative of the loss with respect to the vectors is:

$$\frac{\partial L}{\partial v_{pos}} = (\sigma(v_{pos} \cdot v_c) - 1)v_c$$

$$\frac{\partial L}{\partial v_{neg,i}} = \sigma(v_{neg,i} \cdot v_c)v_c$$

$$\frac{\partial L}{\partial v_c} = (\sigma(v_{pos} \cdot v_c) - 1)v_{pos} + \sum_{i=1}^{k} \sigma(v_{neg,i} \cdot v_c)v_{neg,i}$$



## Project Structure
* `preprocessing.py`: Handles text cleaning, vocabulary building, and unigram table generation.
* `model.py`: Contains the Word2Vec class with forward and backward pass logic.
* `train.py`: The main execution script with the training loop and similarity functions.

## Results(some examples)
*Note: Results are based on 10 epochs of training on the full text.*

| Target Word | Top 5 Closest Neighbors                                                                                                                                                                                                                                           |
| :--- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Alyosha** | [('when', 0.9958031626295162), ('yet', 0.9957935423914862),('even', 0.9955148257453871), ('time', 0.9954859922870285), ('moment', 0.9951211594017303)]                                                                                                            |
| **Ivan** | [('answer', 0.997466282761806), ('asked', 0.9970048072550589), ('question', 0.9965979859213339), ('loved', 0.9965747469996766), ('murder', 0.996337803070265)]                                                                                                    |
| **Zossima** | [('hate', 0.9983219069240433), ('please', 0.9981771957183364), ('fall', 0.9982035791482503), ('carried', 0.9981996810420816), ('gate', 0.9981826815563549)]                                                                                                       |
| **God** | [('yourself', 0.9970252825598657), ('done', 0.9964414630022291), ('get', 0.9960936323869999), ('good', 0.9956421919123472), ('thing', 0.995615131567891)]                                                                                                         |
| **Faith** | [('account', 0.9980125389741913), ('dead', 0.9977484446540933), ('trying', 0.9977056011051022), ('blood', 0.9976039117574856), ('awful', 0.997564939754874)]                                                                                                      |
