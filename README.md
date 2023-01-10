# Calculating `RelationOperator` for the `EleutherAI/gpt-neox-20b` model.

This branch implements calculating `RelationOperator` (Jacobian weights and biases) with the `EleutherAI/gpt-neox-20b` model. 

# Why?
The `NeoX` model occupies 41GB out of 49GB of an A6000 GPU on half precision. To calculate jacobians we need another 41GB of additional memory, which is infeasible with an A6000. <br/>
Usually, we calculate Jacobians from one of the middle layers, say on layer 25 out of 44 layers. So, we create a **smaller** NeoX model by discarding all the initial layers upto some layer (`break_layer_idx` = 22). Then, the input of the first layer of the smaller model is replaced with the output of layer 24 of the original model. The smaller model is roughly half the size of the original model and we can do `backprop()` to populate gradient values downto a particular layer upto $25^{th}$ layer.

## How to use it?
* Go to the `NeoX/split_and_save_weights.ipynb` file. Set the `MODEL_NAME` and on which layer you want to split the model by settin the `break_layer_idx` value. This will create a new folder and the weights of the smaller model will be saved on your machine. **You will only need to run this once**, assuming you will not want to check jacobians for an earlier layer.
* Check the `NeoX/Demo.ipynb` folder. You will want to use `estimate.estimate_relation_operator_neox` method. The interface is basically same as `estimate.estimate_relation_operator`, it just assumes you will pass a model and another smaller model loaded on different GPUs.

## Runtime
Right now it takes about 17-18 minutes to calculate the jacobian `weight` and `bias` for a single `RelationOperator`. <br/>

* Due to memory constraints each rows of the jacobian matrix had to be calculated separately).
* There are some minor differences. Which performing a sanity check with the `gpt2-xl` model, I noticed each row of the jacobian weights were off by an L2 distance of $7.987 \times 10^{-6}$, which I ignore but it was interesting that each row was off by the same value.