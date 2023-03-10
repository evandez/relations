from typing import Any, Sequence, TypeAlias, List

import baukit
from baukit import nethook
import torch
import torch.autograd.functional
import torch.nn
import transformers
import copy
import matplotlib.pyplot as plt

Model: TypeAlias = transformers.GPT2LMHeadModel
ModelInput: TypeAlias = transformers.BatchEncoding
Tokenizer: TypeAlias = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping: TypeAlias = Sequence[tuple[int, int]]
Device: TypeAlias = int | str | torch.device
import warnings

class CornerEstimator:
    """Implements a relation operator for the given LM."""
    model: Model
    tokenizer: Tokenizer


    def __init__(
        self,
        model: Model, tokenizer: Tokenizer,
        ln_f_name: str = "transformer.ln_f", unembedder_module_name: str = "lm_head",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.unembedder = nethook.get_module(model, unembedder_module_name)
        self.ln_f = nethook.get_module(model, ln_f_name)

        self.unembedder_weight_inv = None
        self.unembedder_module_name = unembedder_module_name
        self.ln_f_name = ln_f_name
    

    def get_vocab_representation(
        self, h, 
        perform_layer_norm = True, return_top_k = 5, get_logits = False
    ):
        """
        get representation of vector `h` in the vocabulary space. basically applied logit lens
        """
        z = h.clone()
        if(perform_layer_norm == True and self.ln_f is not None):
            z = self.ln_f(z)
        logits = self.unembedder(z)
        token_ids = logits.topk(dim=-1, k=return_top_k).indices.squeeze().tolist()
        logit_values = logits.topk(dim=-1, k=return_top_k).values.squeeze().tolist()
        return [
            self.tokenizer.decode(t) if get_logits == False else (self.tokenizer.decode(t), round(v, 3))
            for t, v in zip(token_ids, logit_values)
        ]

    
    def estimate_simple_corner(
        self,
        target_words: List[str],
        scale_up = 70,
    ):
        """
        estimates the corner by averaging corresponding rows of the unembedder matrix
        Params:
            target_words:   list of words for which to estimate the corner
            scale_up    :   the estimated corner usually has very small norm. 
                            it needs to multiplied by `scale_up` value to observe the effect.
             
        """
        target_tokenized = self.tokenizer(target_words, padding=True, return_tensors="pt").to(self.model.device)
        interested_rows = torch.stack([self.unembedder.weight[r[0].item()] for r in target_tokenized.input_ids])        
        z = interested_rows.mean(dim = 0)
        return z * scale_up
    
    def estimate_lin_inv_corner(
        self,
        target_words: List[str],
        target_logit_value = 50,
    ):
        """
        logits = W.z + b => z = W.inv() @ (logits - b) = corner
        Params:
            target_words       :  list of words for which to estimate the corner
            target_logit_value :  the desired logit value for each of the target words 
                                  (the actual logit assigned after being processed by the final layer norm and unembedding head is likely to be
                                  much higher. this param basically effects the norm)
        """
        target_tokenized = self.tokenizer(target_words, padding=True, return_tensors="pt").to(self.model.device)
        expected_logit = torch.zeros(self.model.config.vocab_size).to(self.model.dtype).to(self.model.device)
        for t in target_tokenized.input_ids:
            expected_logit[t[0]] = target_logit_value
        
        if (self.unembedder_weight_inv is None):
            print("calculating inverse of unbedding weights . . .")
            if self.model.dtype == torch.float16:
                weight = self.unembedder.weight.to(torch.float32)
                self.unembedder_weight_inv = weight.pinverse().to(self.model.dtype)
            else:
                self.unembedder_weight_inv = self.unembedder.weight.pinverse()

        bias = self.unembedder.bias if self.unembedder.bias is not None else torch.zeros(expected_logit.shape).to(self.model.dtype).to(self.model.device)
        z = self.unembedder_weight_inv @ (expected_logit - bias)
        return z
    
    def estimate_corner_lstsq_solve(
        self,
        target_words: List[str],
        target_logit: int = 50, # the target logit value will not be the logit assigned after 
    ):
        """
        logits = W.z + b
        => W.z = logits - b
        finds z = corner using least square approximation.

        Params:
            target_words       :  list of words for which to estimate the corner
            target_logit_value :  the desired logit value for each of the target words 
                                  (the actual logit assigned after being processed by the final layer norm and unembedding head is likely to be
                                  much higher. this param basically effects the norm)
        """
        # print(target_words)
        target_tokenized = self.tokenizer(target_words, padding=True, return_tensors="pt").to(self.model.device)
        W = torch.stack([self.unembedder.weight[r[0].item()] for r in target_tokenized.input_ids])
        # print(target_tokenized.input_ids.shape, W.shape)
        if (self.unembedder.bias is not None):
            b = self.unembedder.bias[target_tokenized.input_ids]
            b = b.reshape(b.shape[0])
        else:
            b = torch.zeros(len(target_words)).to(self.model.dtype).to(self.model.device)
        y = (torch.ones(len(target_words)) * target_logit).to(self.model.dtype).to(self.model.device) - b
        if(self.model.dtype == torch.float16):
            W = W.to(torch.float32)
            y = y.to(torch.float32)
        x = torch.linalg.lstsq(W, y).solution
        # print(W@x + b)
        return x.to(self.model.dtype)


    def estimate_corner_with_gradient_descent(
        self,
        target_words: List[str],
        target_logit_value: float = 50,
        learning_rate: float = 5e-2,
        weight_decay: float = 2e-2,
        num_steps: int = 100,
        verbose = False,
        k = 1, # penalize the difference values
    ):
        if self.model.dtype == torch.float16:
            warnings.warn(
                """
                model.dtype = torch.float16 ==> applying gradient descent might cause underflow, which will cause
                some values to be divided by 0. Might get `nan` values in the corner
                """
            )

        target_tokenized = self.tokenizer(target_words, padding=True, return_tensors="pt").to(self.model.device)

        tunable_weights = {}
        for n, p in self.model.named_parameters():
            if n.startswith(self.ln_f_name) or n.startswith(self.unembedder_module_name):
                tunable_weights[n] = p
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        z = torch.FloatTensor(self.model.config.n_embd).uniform_(-1.001 , 1.001).to(self.model.dtype).to(self.model.device)
        if(verbose):
            print("initial representation: ", self.get_vocab_representation(z))
        
        z.requires_grad = True
        optimizer = torch.optim.Adam(
            [z],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        loss_track = []
        for iter in range(num_steps):
            logits = self.unembedder(self.ln_f(z))
            target_logits = torch.gather(logits, 0, target_tokenized.input_ids.reshape(len(target_words)))

            optimal_logit_values = torch.zeros(target_logits.shape) + target_logit_value
            optimal_logit_values = optimal_logit_values.to(self.model.dtype).to(self.model.device)
            loss = (optimal_logit_values - target_logits).square().mean() # + logits.mean()
            # mean = target_logits.mean()
            # difference = (target_logits - mean).abs().sum()
            # loss = k*difference - mean # penalize the difference while driving the mean as up as possible
            # print((optimal_logit_values - target_logits).square().mean().item(), logits.mean().item())
            loss_track.append(loss.item())
            # print(loss.item(), logits.mean().item(), target_logits.sum().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for t in tunable_weights:
                tunable_weights[t].grad.zero_()
        
        for t in tunable_weights:
            tunable_weights[t].requires_trad = False
        z.requires_grad = False

        if(verbose):
            plt.rcdefaults()
            plt.plot(loss_track)
            plt.xticks(range(0, len(loss_track), 10))
            plt.xlabel("Iteration")
            plt.ylabel("loss")
            plt.show()

            print("final representation: ", self.get_vocab_representation(z))

        return z

    
    def estimate_average_corner_with_gradient_descent(
        self,
        target_words: List[str],
        average_on: int = 5,
        learning_rate: float = 5e-2,
        weight_decay: float = 2e-2,
        num_steps: int = 100,
        # target_logit_value: float = 50,
        verbose = False
    ):
        corners = [self.estimate_corner_with_gradient_descent(
            target_words, 
            # target_logit_value,
            learning_rate, weight_decay, num_steps, 
            verbose 
        ) for _ in range(average_on)]

        corner = torch.stack(corners)
        return corner.mean(dim = 0)